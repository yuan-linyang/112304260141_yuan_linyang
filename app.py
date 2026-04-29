import os
os.add_dll_directory(r'D:\神经网络\venv\Lib\site-packages\torch\lib')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

def preprocess_image(image):
    try:
        if image is None:
            print("图像为 None")
            return None
        
        # Gradio Sketchpad 可能返回字典，需要提取图像
        if isinstance(image, dict):
            # 从字典中提取图像数据
            if 'composite' in image:
                image_data = image['composite']
            elif 'base' in image:
                image_data = image['base']
            else:
                image_data = list(image.values())[0]
            
            # 如果是 numpy 数组
            if isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data)
            # 如果是 PIL Image
            elif isinstance(image_data, Image.Image):
                image = image_data
            # 如果是 base64 字符串
            elif isinstance(image_data, str):
                import base64
                img_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(img_bytes))
            else:
                print(f"未知的图像数据类型：{type(image_data)}")
                return None
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass  # 已经是 PIL Image
        else:
            print(f"未知的图像类型：{type(image)}")
            return None
        
        print(f"原始图像模式：{image.mode}, 大小：{image.size}")
        
        # 处理 RGBA 图像（移除 alpha 通道）
        if image.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            # 使用 alpha 通道粘贴
            background.paste(image, mask=image.split()[3])
            image = background
            print("已转换 RGBA 为 RGB")
        
        # 转换为灰度图
        if image.mode != 'L':
            image = image.convert('L')
            print("已转换为灰度图")
        
        # 调整大小到 28x28
        image = image.resize((28, 28), Image.BILINEAR)
        image = np.array(image)
        
        print(f"处理后图像形状：{image.shape}, 数值范围：[{image.min()}, {image.max()}]")
        
        # 归一化到 0-1
        if image.max() > 1.0:
            image = image / 255.0
        
        # 反转颜色（MNIST 是黑底白字）
        image = 1.0 - image
        
        # 转换为 PyTorch 张量
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        image = image.to(device)
        
        print(f"张量形状：{image.shape}")
        return image
    except Exception as e:
        print(f"预处理错误：{e}")
        import traceback
        traceback.print_exc()
        return None

def predict_digit(image):
    try:
        if image is None:
            return "请先上传或绘制数字", None, None
        
        processed = preprocess_image(image)
        
        if processed is None:
            return "图像预处理失败", None, None
        
        with torch.no_grad():
            output = model(processed)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probs = probabilities.cpu().numpy()[0]
            prediction = int(torch.argmax(output, dim=1).item())
        
        top_3_indices = np.argsort(probs)[::-1][:3]
        top_3_probs = probs[top_3_indices]
        
        # 格式化为 Gradio JSON 组件可接受的格式
        top_3_results = {
            "predictions": [
                {"digit": int(idx), "probability": f"{float(prob)*100:.2f}%"}
                for idx, prob in zip(top_3_indices, top_3_probs)
            ]
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        digits = [str(i) for i in range(10)]
        colors = ['green' if i == prediction else 'gray' for i in range(10)]
        ax.bar(digits, probs, color=colors, alpha=0.7)
        ax.set_xlabel('Digit', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        for i, v in enumerate(probs):
            ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        prob_plot = Image.open(buf)
        
        confidence = top_3_probs[0]
        result_text = f"🎯 预测结果：数字 {prediction}\n置信度：{confidence*100:.2f}%"
        
        return result_text, top_3_results, prob_plot
    except Exception as e:
        print(f"预测错误：{e}")
        import traceback
        traceback.print_exc()
        return f"识别出错：{str(e)}", None, None

def sketch_to_image(sketch):
    if sketch is None:
        return None
    return sketch

with gr.Blocks(title="MNIST 手写数字识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 MNIST 手写数字识别系统")
    gr.Markdown("### 基于 CNN 的深度学习实践 | 验证准确率：99.35%")
    
    with gr.Tabs():
        with gr.TabItem("📤 上传图片识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 上传手写数字图片")
                    image_input = gr.Image(type="pil", label="上传手写数字图片", height=300)
                    recognize_btn = gr.Button("🔍 识别", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 识别结果")
                    result_text = gr.Textbox(label="预测结果", placeholder="识别结果将显示在这里")
                    top_3_output = gr.JSON(label="🏆 Top 3 预测")
                    prob_plot_output = gr.Image(label="📊 概率分布图")
            
            recognize_btn.click(
                fn=predict_digit,
                inputs=image_input,
                outputs=[result_text, top_3_output, prob_plot_output]
            )
        
        with gr.TabItem("✏️ 在线手写识别"):
            gr.Markdown("### 在下方画板中绘制数字（0-9）")
            
            with gr.Row():
                with gr.Column(scale=1):
                    sketch_input = gr.Sketchpad(label="✍️ 手写画板", type="pil", height=400, width=400,
                                               brush=gr.Brush(colors=["#000000"], color_mode="fixed"))
                    with gr.Row():
                        sketch_recognize_btn = gr.Button("🔍 识别手写数字", variant="primary")
                        clear_btn = gr.Button("🗑️ 清空画板")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 识别结果")
                    sketch_result_text = gr.Textbox(label="预测结果", placeholder="识别结果将显示在这里")
                    sketch_top_3_output = gr.JSON(label="🏆 Top 3 预测")
                    sketch_prob_plot = gr.Image(label="📊 概率分布图")
            
            def recognize_sketch(sketch):
                if sketch is None:
                    return "请先绘制数字", None, None
                return predict_digit(sketch)
            
            sketch_recognize_btn.click(
                fn=recognize_sketch,
                inputs=sketch_input,
                outputs=[sketch_result_text, sketch_top_3_output, sketch_prob_plot]
            )
            
            def clear_canvas():
                return None, "📝 识别结果将显示在这里", {"predictions": []}, None
            
            clear_btn.click(
                fn=clear_canvas,
                inputs=[],
                outputs=[sketch_input, sketch_result_text, sketch_top_3_output, sketch_prob_plot]
            )
    
    gr.Markdown("""
    ---
    ### 📖 使用说明
    1. **上传图片**: 支持 JPG、PNG 等格式的手写数字图片
    2. **在线绘制**: 在画板上绘制数字，支持鼠标或触屏
    3. **识别速度**: 毫秒级响应
    4. **Top-3 预测**: 显示概率最高的 3 个结果及置信度
    
    ### 🧠 模型信息
    - **架构**: CNN (Conv2d + BatchNorm + ReLU + MaxPool)
    - **优化器**: Adam (lr=0.001, weight_decay=1e-5)
    - **数据增强**: Rotation, Translation, Zoom, Gaussian Noise
    - **Early Stopping**: Patience=15
    - **验证准确率**: 99.35%
    - **训练平台**: PyTorch 2.5.1+cu121
    
    ### 👤 学生信息
    - **姓名**: 袁林阳
    - **学号**: 112304260141
    - **班级**: [请填写]
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
