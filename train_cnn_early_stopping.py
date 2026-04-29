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

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
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
print(f"Using device: {device}")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
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
            self.best_model_weights = model.state_dict().copy()
            self.best_epoch = 0
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_weights)
                    print(f'Restored best model weights from epoch {self.best_epoch}')
        else:
            self.best_loss = val_loss
            self.best_model_weights = model.state_dict().copy()
            self.best_epoch = 0
            self.counter = 0

early_stopping = EarlyStopping(patience=15, min_delta=0.0001, restore_best_weights=True)

print("Training CNN with Early Stopping...")
max_epochs = 100
train_losses = []
val_losses = []

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
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
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    
    early_stopping(avg_val_loss, model)
    
    if (epoch + 1) % 5 == 0 or epoch == 0 or early_stopping.early_stop:
        print(f'Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}!")
        print(f"Best validation loss: {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch + 1}")
        break

actual_epochs = epoch + 1
print(f"\nTraining completed! Trained for {actual_epochs} epochs (max: {max_epochs})")

plt.figure(figsize=(12, 8))
plt.plot(range(1, actual_epochs + 1), train_losses[:actual_epochs], 'b-', linewidth=2, label='Train Loss')
plt.plot(range(1, actual_epochs + 1), val_losses[:actual_epochs], 'r-', linewidth=2, label='Validation Loss')

best_epoch_marker = early_stopping.best_epoch + 1
plt.axvline(x=best_epoch_marker, color='g', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch_marker})')
plt.axhline(y=early_stopping.best_loss, color='orange', linestyle=':', linewidth=1.5, label=f'Best Val Loss ({early_stopping.best_loss:.4f})')

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title(f'CNN Training with Early Stopping (Stopped at Epoch {actual_epochs})', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, actual_epochs + 1, max(1, actual_epochs // 10)))
plt.tight_layout()

plt.savefig(os.path.join(script_dir, 'loss_curve_early_stopping.png'), dpi=300, bbox_inches='tight')
print("Loss curve saved as 'loss_curve_early_stopping.png'")

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
print(f"Submission file saved! Total predictions: {len(predictions)}")
print(f"First 10 predictions: {predictions[:10]}")
print(f"Prediction distribution: {np.bincount(predictions)}")
print(f"\nEarly Stopping Summary:")
print(f"  - Patience: {early_stopping.patience} epochs")
print(f"  - Best Epoch: {best_epoch_marker}")
print(f"  - Best Validation Loss: {early_stopping.best_loss:.4f}")
print(f"  - Total Epochs Trained: {actual_epochs}")
print(f"  - Epochs Saved: {max_epochs - actual_epochs}")
