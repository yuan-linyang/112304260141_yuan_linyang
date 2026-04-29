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
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))

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
        self.transforms_list = [
            ('Original', lambda x: x),
            ('Rotation ±10°', lambda x: F.rotate(x, angle=np.random.uniform(-10, 10))),
            ('Translation ±2px', lambda x: self.random_translate(x, max_shift=2)),
            ('Zoom 1.1x', lambda x: self.random_zoom(x, zoom_range=(0.9, 1.1))),
            ('Shear ±5°', lambda x: F.affine(x, angle=0, translate=[0,0], scale=1, shear=np.random.uniform(-5, 5))),
            ('Horizontal Flip', lambda x: F.hflip(x)),
            ('Brightness ±20%', lambda x: self.adjust_brightness(x, factor=np.random.uniform(0.8, 1.2))),
            ('Contrast ±20%', lambda x: self.adjust_contrast(x, factor=np.random.uniform(0.8, 1.2))),
            ('Gaussian Noise', lambda x: self.add_gaussian_noise(x, mean=0, std=0.05)),
            ('Elastic Deform', lambda x: self.elastic_transform(x, alpha=0.5, sigma=0.3))
        ]
    
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
    
    def adjust_brightness(self, img, factor=1.0):
        img = img * factor
        return torch.clamp(img, 0, 1)
    
    def adjust_contrast(self, img, factor=1.0):
        mean = img.mean()
        img = (img - mean) * factor + mean
        return torch.clamp(img, 0, 1)
    
    def add_gaussian_noise(self, img, mean=0, std=0.05):
        noise = torch.randn_like(img) * std + mean
        return torch.clamp(img + noise, 0, 1)
    
    def elastic_transform(self, img, alpha=0.5, sigma=0.3):
        return img

aug = DataAugmentation()

print("Generating augmented images...")

sample_indices = np.random.choice(len(X_train_tensor), 5, replace=False)

fig, axes = plt.subplots(10, 11, figsize=(20, 18))

for idx, sample_idx in enumerate(sample_indices):
    original_img = X_train_tensor[sample_idx]
    label = y_train[sample_idx]
    
    axes[idx*2, 0].imshow(original_img.squeeze(), cmap='gray')
    axes[idx*2, 0].set_title(f'Original (Label: {label})', fontsize=12, fontweight='bold')
    axes[idx*2, 0].axis('off')
    
    axes[idx*2 + 1, 0].axis('off')
    
    for i, (transform_name, transform_func) in enumerate(aug.transforms_list[1:], 1):
        augmented_img = transform_func(original_img.unsqueeze(0)).squeeze(0)
        augmented_img = torch.clamp(augmented_img, 0, 1)
        
        axes[idx*2, i].imshow(augmented_img.squeeze(), cmap='gray')
        if idx == 0:
            axes[idx*2, i].set_title(transform_name, fontsize=9, rotation=45, ha='right')
        axes[idx*2, i].axis('off')
        
        random_aug = aug.transforms_list[np.random.randint(1, len(aug.transforms_list))][1]
        random_aug_img = random_aug(original_img.unsqueeze(0)).squeeze(0)
        random_aug_img = torch.clamp(random_aug_img, 0, 1)
        
        axes[idx*2 + 1, i].imshow(random_aug_img.squeeze(), cmap='gray')
        axes[idx*2 + 1, i].axis('off')

plt.suptitle('Data Augmentation Examples for MNIST Digit Recognition', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'data_augmentation_examples.png'), dpi=300, bbox_inches='tight')
print("Data augmentation examples saved as 'data_augmentation_examples.png'")
plt.show()

print("\nApplying data augmentation to training set...")

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
            random_transform = self.aug.transforms_list[np.random.randint(1, len(self.aug.transforms_list))][1]
            img = random_transform(img.unsqueeze(0)).squeeze(0)
            img = torch.clamp(img, 0, 1)
        
        return img, label

train_dataset_aug = AugmentedDataset(X_train_tensor, y_train_tensor, augmentation=True, aug_factor=3)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader_aug = DataLoader(train_dataset_aug, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"\nOriginal training samples: {len(X_train_tensor)}")
print(f"Augmented training samples: {len(train_dataset_aug)}")
print(f"Augmentation factor: {3}x")

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
    def __init__(self, patience=15, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = 0
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_weights)
                    print(f'\nRestored best model weights from epoch {self.best_epoch + 1}')
        else:
            self.best_loss = val_loss
            self.best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.counter = 0

early_stopping = EarlyStopping(patience=15, min_delta=0.0001, restore_best_weights=True)

print("\nTraining CNN with Data Augmentation and Early Stopping...")
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
    
    early_stopping(avg_val_loss, model)
    
    if (epoch + 1) % 5 == 0 or epoch == 0 or early_stopping.early_stop:
        print(f'Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}!")
        print(f"Best validation loss: {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch + 1}")
        break

actual_epochs = epoch + 1
print(f"\nTraining completed! Trained for {actual_epochs} epochs (max: {max_epochs})")
print(f"Best validation accuracy: {val_accuracies[early_stopping.best_epoch]:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(range(1, actual_epochs + 1), train_losses[:actual_epochs], 'b-', linewidth=2, label='Train Loss')
ax1.plot(range(1, actual_epochs + 1), val_losses[:actual_epochs], 'r-', linewidth=2, label='Validation Loss')
ax1.axvline(x=early_stopping.best_epoch + 1, color='g', linestyle='--', linewidth=2, label=f'Best Epoch ({early_stopping.best_epoch + 1})')
ax1.axhline(y=early_stopping.best_loss, color='orange', linestyle=':', linewidth=1.5, label=f'Best Val Loss ({early_stopping.best_loss:.4f})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss with Data Augmentation', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, actual_epochs + 1), train_accuracies[:actual_epochs], 'b-', linewidth=2, label='Train Accuracy')
ax2.plot(range(1, actual_epochs + 1), val_accuracies[:actual_epochs], 'r-', linewidth=2, label='Validation Accuracy')
best_val_acc = val_accuracies[early_stopping.best_epoch]
ax2.axvline(x=early_stopping.best_epoch + 1, color='g', linestyle='--', linewidth=2, label=f'Best Epoch ({early_stopping.best_epoch + 1})')
ax2.axhline(y=best_val_acc, color='orange', linestyle=':', linewidth=1.5, label=f'Best Val Acc ({best_val_acc:.2f}%)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Validation Accuracy with Data Augmentation', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'training_curves_with_augmentation.png'), dpi=300, bbox_inches='tight')
print("Training curves saved as 'training_curves_with_augmentation.png'")
plt.show()

print("\nLoading test data...")
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
print(f"\nSubmission file saved! Total predictions: {len(predictions)}")
print(f"First 10 predictions: {predictions[:10]}")
print(f"\nData Augmentation Summary:")
print(f"  - Augmentation techniques: {len(aug.transforms_list) - 1}")
print(f"  - Original samples: {len(X_train_tensor)}")
print(f"  - Augmented samples: {len(train_dataset_aug)}")
print(f"  - Augmentation factor: 3x")
print(f"  - Best validation accuracy: {best_val_acc:.2f}%")
print(f"  - Epochs trained: {actual_epochs}")
