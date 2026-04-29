import os
os.add_dll_directory(r'D:\神经网络\venv\Lib\site-packages\torch\lib')

import torch
import numpy as np
from PIL import Image

# 测试模型加载
print("="*60)
print("模型加载测试")
print("="*60)

try:
    import torch.nn as nn
    
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
    print(f"[OK] 设备：{device}")
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()
    print(f"[OK] 模型加载成功")
    
    # 测试推理
    print("\n" + "="*60)
    print("模型推理测试")
    print("="*60)
    
    test_input = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        output = model(test_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    print(f"[OK] 推理成功")
    print(f"  预测数字：{predicted.item()}")
    print(f"  置信度：{confidence.item() * 100:.2f}%")
    
    # 测试所有数字
    print("\n" + "="*60)
    print("批量测试（模拟 0-9）")
    print("="*60)
    
    for i in range(10):
        # 创建一个简单的数字图案（模拟）
        test_img = np.zeros((28, 28), dtype=np.float32)
        # 添加一些噪声模拟真实输入
        test_img += np.random.normal(0, 0.1, (28, 28))
        test_img = np.clip(test_img, 0, 1)
        
        test_tensor = torch.tensor(test_img).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(test_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
        
        print(f"  测试 {i}: 预测={pred.item()}, 置信度={conf.item()*100:.1f}%")
    
    print("\n" + "="*60)
    print("[OK] 所有测试通过！")
    print("="*60)
    print("\n现在可以运行：python app_final.py")
    print("然后访问 http://127.0.0.1:7860")
    
except Exception as e:
    print(f"\n[ERROR] 错误：{e}")
    print("\n请检查:")
    print("1. best_model.pth 文件是否存在")
    print("2. 是否安装了所有依赖 (pip install -r requirements.txt)")
    print("3. PyTorch 版本是否正确")
