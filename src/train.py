import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import ImageFolder

# 移除損壞的圖片

def remove_corrupted_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # 驗證圖片文件是否完整
            except (IOError, SyntaxError, UnidentifiedImageError) as e:
                print(f"移除損壞的圖片: {file_path}")
                os.remove(file_path)  # 刪除損壞的文件

# 自定義的 SafeImageFolder 類來跳過損壞圖片

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, IOError) as e:
            print(f"跳過損壞的圖片: {self.imgs[index][0]}")
            return self[index + 1]  # 遞歸地返回下一個

# 資料增強和數據處理
data_dir = 'data/processed'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# 刪除損壞的圖片
remove_corrupted_images(train_dir)
remove_corrupted_images(val_dir)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 使用自定義的 SafeImageFolder 來加載圖片
train_dataset = SafeImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = SafeImageFolder(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 使用 ResNet18 作為遷移學習模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 類別 (貓和狗)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 記錄損失和準確度
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader:
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                except Exception as e:
                    print(f"跳過損壞的圖片: {e}")

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 繪製訓練和驗證的損失與準確度
    plot_training_results()

def plot_training_results():
    epochs = range(1, len(train_loss_history) + 1)

    # 繪製損失曲線
    plt.figure()
    plt.plot(epochs, train_loss_history, 'r', label='Training loss')
    plt.plot(epochs, val_loss_history, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 繪製準確度曲線
    plt.figure()
    plt.plot(epochs, train_acc_history, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc_history, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)