import os
os.add_dll_directory(r'D:\神经网络\venv\Lib\site-packages\torch\lib')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("实验 4: Adam + Early Stopping + 数据增强 + Batch Size 64")
print("="*60)

print("Loading data...")
train_df = pd.read_csv(os.path.join(script_dir, 'train.csv'))

X = train_df.drop('label', axis=1).values.astype(np.float32) / 255.0
y = train_df['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train).reshape(-1, 1, 28, 28)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val).reshape(-1, 1, 28, 28)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

class DataAugmentation:
    def __init__(self):
        pass
    
    def random_rotate(self, img, angle=10):
        return F.rotate(img, angle=np.random.uniform(-angle, angle))
    
    def random_translate(self, img, max_shift=2):
        h, w = img.shape[-2:]
        tx = np.random.randint(-max_shift, max_shift+1)
        ty = np.random.randint(-max_shift, max_shift+1)
        return F.affine(img, angle=0, translate=[tx, ty], scale=1, shear=[0, 0])
    
    def random_zoom(self, img, zoom_range=(0.9, 1.1)):
        zoom_factor = np.random.uniform(*zoom_range)
        new_size = int(28 * zoom_factor)
        resized = F.resize(img, [new_size, new_size])
        return F.resize(resized, [28, 28])
    
    def add_gaussian_noise(self, img, std=0.05):
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0, 1)
    
    def augment(self, img):
        img = self.random_rotate(img, 10)
        img = self.random_translate(img, 2)
        img = self.random_zoom(img)
        if np.random.random() > 0.5:
            img = self.add_gaussian_noise(img, 0.05)
        return img

aug = DataAugmentation()

class AugmentedDataset:
    def __init__(self, X_tensor, y_tensor, augmentation=True, aug_factor=3):
        self.X_tensor = X_tensor
        self.y_tensor = y_tensor
        self.augmentation = augmentation
        self.aug_factor = aug_factor
        self.aug = DataAugmentation()
    
    def __len__(self):
        if self.augmentation:
            return len(self.X_tensor) * self.aug_factor
        return len(self.X_tensor)
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.X_tensor)
        img = self.X_tensor[original_idx]
        label = self.y_tensor[original_idx]
        
        if self.augmentation and idx >= len(self.X_tensor):
            img = self.aug.augment(img.unsqueeze(0)).squeeze(0)
            img = torch.clamp(img, 0, 1)
        
        return img, label

train_dataset_aug = AugmentedDataset(X_train_tensor, y_train_tensor, augmentation=True, aug_factor=3)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader_aug = DataLoader(train_dataset_aug, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"\nOriginal training samples: {len(X_train_tensor)}")
print(f"Augmented training samples: {len(train_dataset_aug)}")
print(f"Augmentation factor: 3x")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_model_weights)
                print(f'\nRestored best model weights from epoch {self.best_epoch + 1}')
        else:
            self.best_loss = val_loss
            self.best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.counter = 0

early_stopping = EarlyStopping(patience=15, min_delta=0.0001)

print("\nTraining with Adam + Data Augmentation + Early Stopping...")
max_epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader_aug:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    avg_train_loss = train_loss / len(train_loader_aug)
    train_acc = 100 * correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    
    early_stopping(avg_val_loss, model, epoch)
    
    if (epoch + 1) % 5 == 0 or epoch == 0 or early_stopping.early_stop:
        print(f'Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}!")
        print(f"Best validation loss: {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch + 1}")
        break

actual_epochs = epoch + 1
best_val_acc = val_accuracies[early_stopping.best_epoch]
print(f"\nTraining completed! Trained for {actual_epochs} epochs")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

results = {
    'experiment': 'Exp4_Adam_ES_Aug_BS64',
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'batch_size': 64,
    'data_augmentation': True,
    'early_stopping': True,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'best_val_acc': best_val_acc,
    'best_epoch': early_stopping.best_epoch,
    'actual_epochs': actual_epochs
}

np.save(os.path.join(script_dir, 'exp4_results.npy'), results, allow_pickle=True)
print("Results saved to exp4_results.npy")

torch.save(model.state_dict(), os.path.join(script_dir, 'best_model.pth'))
print("Best model saved to best_model.pth")

print("\nGenerating predictions for test set...")
test_df = pd.read_csv(os.path.join(script_dir, 'test.csv'))
X_test = test_df.values.astype(np.float32) / 255.0
X_test_tensor = torch.tensor(X_test).reshape(-1, 1, 28, 28).to(device)

model.eval()

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    predictions = predicted.cpu().numpy()

submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})
submission.to_csv(os.path.join(script_dir, 'sample_submission.csv'), index=False)
print(f"Submission file saved! Total predictions: {len(predictions)}")

print(f"\n{'='*60}")
print(f"实验 4 完成总结:")
print(f"  - 数据增强技术：Rotation, Translation, Zoom, Noise")
print(f"  - Original samples: {len(X_train_tensor)}")
print(f"  - Augmented samples: {len(train_dataset_aug)}")
print(f"  - Best validation accuracy: {best_val_acc:.2f}%")
print(f"  - Epochs trained: {actual_epochs}")
print(f"{'='*60}")
