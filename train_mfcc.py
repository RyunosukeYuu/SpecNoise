import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from Dataset_loader import BirdSpectrogramLoader, BirdSoundLoader
from data_aug.randomReplacementDropStripes import RandomReplacementSpecAugmentation
from pytorch_utils import do_mixup, random_local_masking, dynamic_background_simulation
from data_aug.specAugmentation import SpecAugmentation
from models import Singlebranchspectralclassifier
import os
import time
from pytorch_utils import specmix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='D:/WMWB/raw', help="Path to the dataset directory")
    parser.add_argument('-b', '--batch_size', default=32, type=int, help="Batch size for training")
    parser.add_argument('-d', '--device', default='cuda', help="Device for training (default: cuda)")
    parser.add_argument('-t', '--train_epochs', default=200, type=int,
                        help="Number of epochs for training (default: 50)")
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float,
                        help="Learning rate for training (default: 0.001)")
    parser.add_argument('-c', '--checkpoint_dir',
                        default='E:/bird_sound_classification/BirdMLClassification-main/SSL_Net/models/audio/MFCC_resnet18/',
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('-a', '--augmentation', nargs='*',
                        choices=['none', 'mixup', 'specaugment', 'random_local_masking',
                                 'dynamic_background_simulation', 'RandomReplacementSpecAugmentation','specmix'],
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
    model = Singlebranchspectralclassifier(num_classes=20)
    model.to(device)
    return model


def load_data(input_path, batch_size, seed):
    print("[INFO] Loading dataset...")
    data_loader = BirdSoundLoader(input_path=input_path, test=0.2, random_seed=seed, feature_type='mfcc')
    trainX, trainY, testX, testY = data_loader.load_and_split()

    trainX, trainY = torch.tensor(trainX).float(), torch.tensor(trainY)
    testX, testY = torch.tensor(testX).float(), torch.tensor(testY)

    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(testX, testY), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, args):
    print("[INFO] Training MLP...")

    spec_augmenter = SpecAugmentation(time_drop_width=1, time_stripes_num=2,
                                      freq_drop_width=4,
                                      freq_stripes_num=2) if 'specaugment' in args.augmentation else None

    aug = RandomReplacementSpecAugmentation(time_drop_width=1, time_stripes_num=2, freq_drop_width=4,
                                            freq_stripes_num=2, replacement='noise')

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_images = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if spec_augmenter:
                inputs = spec_augmenter(inputs)

            if 'mixup' in args.augmentation:
                inputs, y_a, y_b, lam = do_mixup(inputs, labels, alpha=0.2)
                inputs, y_a, y_b = inputs.to(device), y_a.to(device), y_b.to(device)

            if 'random_local_masking' in args.augmentation:
                inputs = random_local_masking(inputs, 16)
            if 'dynamic_background_simulation' in args.augmentation:
                inputs = dynamic_background_simulation(inputs)

            if 'RandomReplacementSpecAugmentation' in args.augmentation:
                inputs = aug(inputs)

            if 'specmix' in args.augmentation:
                inputs, labels = specmix(inputs, labels, 0.2, 2, 4, 4, 4)
                inputs, labels = inputs.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = model(inputs)

            if 'mixup' in args.augmentation:
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(outputs.float(), labels.long())

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
            f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%')

    return {'train_loss': train_loss, 'test_loss': test_loss, 'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy}


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
    train_loader, test_loader = load_data(args.input, args.batch_size, args.seed)
    model = load_model(args.device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=0, last_epoch=-1)

    final_epoch_info = train(model, train_loader, test_loader, criterion, optimizer, scheduler, args.device,
                             args.train_epochs, args)
    save_model(model, args.checkpoint_dir, final_epoch_info, args)