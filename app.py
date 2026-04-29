import os
os.add_dll_directory(r'D:\神经网络\venv\Lib\site-packages\torch\lib')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

def predict(image):
    if image is None:
        return None, "请上传图片或手写数字"
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_digit = predicted.item()
    confidence_score = confidence.item() * 100
    
    prob_array = probabilities.cpu().numpy()[0]
    
    result_text = f"预测结果：{predicted_digit}\n置信度：{confidence_score:.2f}%"
    
    return prob_array, result_text

with gr.Blocks(title="MNIST 手写数字识别") as demo:
    gr.Markdown("# 🎯 MNIST 手写数字识别系统")
    gr.Markdown("请上传一张手写数字图片（28x28 像素，灰度图），或者使用下方的画板手写输入。")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传手写数字图片")
            submit_btn = gr.Button("🔍 开始识别", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="预测结果")
            result_text = gr.Textbox(label="识别结果", lines=2)
    
    gr.Examples(
        examples=[
            ["examples/0.png"],
            ["examples/1.png"],
            ["examples/2.png"],
        ],
        inputs=input_image
    )
    
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_label, result_text]
    )

if __name__ == "__main__":
    demo.launch()
