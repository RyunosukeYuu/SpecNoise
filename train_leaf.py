import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from Dataset_loader import BirdSoundLoader
from models import ResNet18, LEAF, AudioClassifier
from data_aug.data_augment import Mixup, Cutout, CutMix, SpecAugment, SpecFrequencyMask
import os
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='D:/WMWB/raw', help="Path to the dataset directory")
    parser.add_argument('-b', '--batch_size', default=64, type=int, help="Batch size for training")
    parser.add_argument('-d', '--device', default='cuda', help="Device for training (default: cuda)")
    parser.add_argument('-t', '--train_epochs', default=50, type=int,
                        help="Number of epochs for training (default: 50)")
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float,
                        help="Learning rate for training (default: 0.001)")
    parser.add_argument('-c', '--checkpoint_dir',
                        default='E:/bird_sound_classification/BirdMLClassification-main/SSL_Net/models/audio/LEAF_resnet18/',
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('-a', '--augmentation', nargs='*',
                        choices=['none', 'mixup', 'specaugment', 'cutout', 'cutmix', 'specfrequencymask'],
                        default=['none'],
                        help="Data augmentation methods to use (default: none)")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model(device):
    print("[INFO] Loading model...")
    leaf = LEAF()
    resnet18 = ResNet18(num_classes=20)
    model = AudioClassifier(leaf, resnet18)
    model.to(device)
    return model


def load_data(input_path, batch_size, seed, augmentations=None):
    print("[INFO] Loading dataset...")
    data_loader = BirdSoundLoader(input_path, test=0.2, random_seed=seed)
    trainX, trainY, testX, testY = data_loader.load_and_split()
    print(f"[INFO] Train set size: {len(trainX)}, Test set size: {len(testX)}")

    # Convert to tensors and transpose
    trainX, trainY = map(torch.tensor, (trainX, trainY))
    testX, testY = map(torch.tensor, (testX, testY))
    print(f"[INFO] Train set shape: {trainX.shape}, Test set shape: {testX.shape}")

    # Apply SpecAugment to training data
    if 'specaugment' in augmentations:
        specaugment = SpecAugment(time_mask_width=10, num_time_masks=1, freq_mask_width=10, num_freq_masks=1)
        trainX = torch.stack([torch.tensor(specaugment.transform(x.numpy())) for x in trainX])

    if 'specfrequencymask' in augmentations:
        freq_mask = SpecFrequencyMask()
        trainX = np.array([freq_mask.apply(x.numpy()) for x in trainX])
        trainX = torch.tensor(trainX).float()

    # Convert tensors to datasets
    train_dataset = TensorDataset(trainX.float(), trainY)
    test_dataset = TensorDataset(testX.float(), testY)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, augmentations=None):
    print("[INFO] Training MLP...")
    mixup = Mixup(alpha=0.5) if 'mixup' in augmentations else None
    cutout = Cutout(n_holes=2, length=32) if 'cutout' in augmentations else None
    cutmix = CutMix(alpha=0.1) if 'cutmix' in augmentations else None

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_images = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if cutout:
                # Apply Cutout
                inputs = torch.stack([cutout.cutout(img) for img in inputs])

            if mixup:
                # Apply Mixup
                inputs, targets_a, targets_b, lam = mixup.mixup_data(inputs, labels)
                outputs = model(inputs)
                loss = mixup.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif cutmix:
                # Apply CutMix
                inputs, targets_a, targets_b, lam = cutmix.cutmix_data(inputs, labels)
                outputs = model(inputs)
                loss = cutmix.cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                # No data augmentation
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_images

        test_loss, test_correct, test_images = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
                test_images += labels.size(0)

        test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_images
        scheduler.step()

        print(
            f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')

    return {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }


def save_model(model, checkpoint_dir, final_epoch_info, args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ensure_dir(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f'output_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    history_path = os.path.join(checkpoint_dir, f'history_training_{timestamp}.json')
    with open(history_path, 'w') as outfile:
        json.dump({'final_epoch_info': final_epoch_info, 'args': vars(args)}, outfile)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Ensure checkpoint directory exists
    ensure_dir(args.checkpoint_dir)

    print("[INFO] Training parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Load data with optional augmentation
    train_loader, test_loader = load_data(args.input, args.batch_size, args.seed, args.augmentation)
    model = load_model(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=0, last_epoch=-1)

    final_epoch_info = train(model, train_loader, test_loader, criterion, optimizer, scheduler, args.device,
                             args.train_epochs, args.augmentation)
    save_model(model, args.checkpoint_dir, final_epoch_info, args)
