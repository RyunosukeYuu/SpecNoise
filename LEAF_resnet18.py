import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import timm
import json
from transfer_learning.custom_dataset_loader_and_splitter import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='D:/WMWB/raw', help="path to the dataset directory")
parser.add_argument('-b', '--batch_size', default=64, type=int, help="batch size for training")
parser.add_argument('-d', '--device', default='cuda', help="device for training (default: cuda)")
parser.add_argument('-w', '--warmup_epochs', default=20, type=int, help="number of epochs for warmup (default: 25)")
parser.add_argument('-t', '--train_epochs', default=50, type=int, help="number of epochs for training (default: 50)")
parser.add_argument('-x', '--warm_up_lr', default=0.001, type=float, help="learning rate for warmup (default: 0.1)")
parser.add_argument('-c', '--checkpoint_dir',
                    default='E:/bird_sound_classfication/BirdMLClassification-main/SSL_Net/models/audio/LEAF_resnet18/',
                    help="directory to save checkpoints (default: checkpoints)")

args = parser.parse_args()

print("[INFO] loading dataset...")
data_loader = CustomDatasetLoaderAndSplitter_seq(args.input, validation=0.2, test=0.0, random_seed=100)
trainX, trainY, valX, valY = data_loader.load_and_split()
trainX, trainY = map(torch.tensor, (trainX, trainY))
valX, valY = map(torch.tensor, (valX, valY))
trainX = trainX.unsqueeze(1).float()
valX = valX.unsqueeze(1).float()
train_dataset = TensorDataset(trainX.float(), trainY)
val_dataset = TensorDataset(valX.float(), valY)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

print("[INFO] loading model...")

leaf = LEAF()
model = AudioClassifier(frontend=leaf)

model.to(args.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.warm_up_lr)

history_warmup = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

print("[INFO] fine-tuning model...")
history_finetune = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
for epoch in range(args.train_epochs):
    model.train()
    total_loss, total_correct, total_images = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)
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
    history_finetune['train_loss'].append(train_loss)
    history_finetune['train_accuracy'].append(train_accuracy)

    val_loss, val_correct, val_images = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_images += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_images
    history_finetune['val_loss'].append(val_loss)
    history_finetune['val_accuracy'].append(val_accuracy)
    # scheduler.step()

    print(
        f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%')

model_path = args.checkpoint_dir + 'output.pth'
torch.save(model.state_dict(), model_path)
history_training_path = args.checkpoint_dir + 'history_training.txt'
with open(history_training_path, 'w') as outfile:
    json.dump(history_finetune, outfile)

print("[INFO] evaluating after fine-tuning...")
# model.eval()
# with torch.no_grad():
#     correct = total = 0
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(args.device), labels.to(args.device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f'Accuracy: {100 * correct / total:.2f}%')

print("[INFO] serializing model...")
