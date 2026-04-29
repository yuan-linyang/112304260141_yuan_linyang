import os
os.add_dll_directory(r'D:\神经网络\venv\Lib\site-packages\torch\lib')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import gradio as gr
import json
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 0, 1, 2
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 3, 4, 5
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 6, 7, 8
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

model = SimpleCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model.eval()

def preprocess_image(image):
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('L')
    image = ImageOps.invert(image)
    
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).to(device)
    
    return image_tensor

def predict(image):
    if image is None:
        return None, "请绘制或上传数字", []
    
    image_tensor = preprocess_image(image)
    if image_tensor is None:
        return None, "无法处理图片", []
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_digit = predicted.item()
    confidence_score = confidence.item() * 100
    
    prob_array = probabilities.cpu().numpy()[0]
    
    top_probs = []
    for i in range(10):
        top_probs.append({
            'digit': i,
            'probability': float(prob_array[i] * 100)
        })
    
    top_probs.sort(key=lambda x: x['probability'], reverse=True)
    top_3 = top_probs[:3]
    
    result_text = f"预测结果：{predicted_digit}\n置信度：{confidence_score:.2f}%"
    
    return {i: float(prob_array[i] * 100) for i in range(10)}, result_text, top_3

def recognize_from_sketch(sketch_data):
    if sketch_data is None:
        return None, "请在画板上绘制数字", []
    
    return predict(sketch_data)

def recognize_from_upload(image):
    if image is None:
        return None, "请上传图片", []
    
    return predict(image)

def clear_history():
    return [], "历史记录已清空"

with gr.Blocks(title="手写数字识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 手写数字识别系统
    
    基于 CNN 的在线识别页面，支持上传图片、网页手写板、Top-3 预测、概率分布和连续识别历史。
    
    ---
    """)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("🖼️ 上传图片识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 步骤 1: 上传图片")
                    upload_image = gr.Image(type="pil", label="上传手写数字图片", height=300)
                    upload_btn = gr.Button("🔍 识别上传的图片", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 步骤 2: 查看结果")
                    upload_label = gr.Label(num_top_classes=3, label="预测结果")
                    upload_result = gr.Textbox(label="识别详情", lines=2, value="等待识别...")
        
        with gr.TabItem("✏️ 在线手写识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 步骤 1: 绘制数字")
                    sketchpad = gr.Sketchpad(
                        label="手写画板",
                        type="pil",
                        height=300,
                        brush=gr.Brush(colors=["#000000"], default_size=15),
                        canvas_size=(280, 280),
                        interactive=True
                    )
                    with gr.Row():
                        sketch_btn = gr.Button("🔍 识别手写数字", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ 清空画板", variant="secondary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 步骤 2: 查看结果")
                    sketch_label = gr.Label(num_top_classes=3, label="预测结果")
                    sketch_result = gr.Textbox(label="识别详情", lines=2, value="等待识别...")
            
            sketch_btn.click(
                fn=recognize_from_sketch,
                inputs=sketchpad,
                outputs=[sketch_label, sketch_result]
            )
            
            clear_btn.click(
                fn=lambda: None,
                inputs=None,
                outputs=sketchpad
            )
        
        upload_btn.click(
            fn=recognize_from_upload,
            inputs=upload_image,
            outputs=[upload_label, upload_result]
        )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 使用说明")
            gr.Markdown("""
            1. **上传图片**: 支持 JPG、PNG 等格式
            2. **手写输入**: 在画板上绘制数字后点击识别
            3. **清空画板**: 清除当前绘制内容
            4. **Top-3 预测**: 显示概率最高的 3 个数字
            5. **置信度**: 显示预测的可靠程度
            
            **提示**: 
            - 绘制时尽量使数字居中
            - 使用黑色画笔，白色背景
            - 数字大小适中，不要超出画板
            """)
        
        with gr.Column():
            gr.Markdown("### ℹ️ 模型信息")
            gr.Markdown("""
            - **模型类型**: 卷积神经网络 (CNN)
            - **训练数据**: MNIST 手写数字数据集
            - **验证准确率**: 99.35%
            - **输入尺寸**: 28×28 像素
            - **输出类别**: 0-9 共 10 个数字
            
            **技术栈**:
            - PyTorch 2.5.1
            - Gradio 3.0+
            - CUDA 12.1 (GPU 加速)
            """)
    
    gr.Markdown("---")
    gr.Markdown("""
    <div style="text-align: center; color: #888; font-size: 12px;">
    <b>实验项目</b> | 基于 CNN 的手写数字识别系统 | 2026
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
