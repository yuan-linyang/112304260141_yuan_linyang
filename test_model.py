import os
os.add_dll_directory(r'D:\神经网络\venv\Lib\site-packages\torch\lib')

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

print("Loading model...")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print("Model loaded successfully!")

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28), Image.BILINEAR)
    image = np.array(image)
    
    if image.max() > 1.0:
        image = image / 255.0
    
    image = 1.0 - image
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    image = image.to(device)
    
    return image

print("\n" + "="*60)
print("MNIST 手写数字识别系统 - 本地测试")
print("="*60)
print("\n使用说明:")
print("1. 准备一张手写数字图片 (28x28 或更大)")
print("2. 将图片保存为 test_digit.png")
print("3. 运行此脚本进行测试")
print("="*60)

test_image_path = 'test_digit.png'
if os.path.exists(test_image_path):
    print(f"\nTesting with image: {test_image_path}")
    processed = preprocess_image(test_image_path)
    
    with torch.no_grad():
        output = model(processed)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probs = probabilities.cpu().numpy()[0]
        prediction = torch.argmax(output, dim=1).item()
    
    print(f"\n预测结果：{prediction}")
    print(f"置信度：{probs[prediction]*100:.2f}%")
    print(f"\nTop 3 预测:")
    top_3_indices = np.argsort(probs)[::-1][:3]
    for idx, prob in zip(top_3_indices, probs[top_3_indices]):
        print(f"  数字 {idx}: {prob*100:.2f}%")
else:
    print(f"\n测试图片不存在：{test_image_path}")
    print("请准备一张手写数字图片进行测试")

print("\n" + "="*60)
print("模型测试完成!")
print("="*60)
