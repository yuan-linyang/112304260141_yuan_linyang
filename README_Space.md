---
title: MNIST 手写数字识别系统
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🎯 MNIST 手写数字识别系统

基于 CNN 的深度学习手写数字识别系统，支持上传图片识别和在线手写板输入。

## ✨ 功能特点

### 实验二：Web 应用
- 🖼️ **图片上传识别**: 支持 JPG、PNG 等格式
- 🎯 **实时预测**: 毫秒级响应
- 📊 **Top-3 预测**: 显示概率最高的 3 个结果
- 💯 **置信度显示**: 精确到小数点后 2 位

### 实验三：交互系统（加分）
- ✏️ **网页手写板**: 交互式画板，支持鼠标/触屏绘制
- 🔄 **连续识别**: 支持多次识别
- 📈 **概率分布图**: 可视化显示 0-9 的概率
- 🗑️ **清空画板**: 一键清除，重新绘制

## 🧠 模型信息

| 项目 | 值 |
|------|-----|
| **架构** | CNN (Conv2d + BatchNorm + ReLU + MaxPool) |
| **优化器** | Adam (lr=0.001, weight_decay=1e-5) |
| **Batch Size** | 64 |
| **数据增强** | Rotation, Translation, Zoom, Noise |
| **Early Stopping** | Patience=15 |
| **验证准确率** | 99.35% |
| **训练集准确率** | 99.70% |
| **参数量** | 1.57M |
| **推理时间** | <50ms (CPU) |

## 🚀 本地运行

```bash
# 1. 克隆仓库
git clone https://huggingface.co/spaces/yuan-linyang/mnist-digit-recognition
cd mnist-digit-recognition

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python app.py

# 4. 访问应用
# 打开浏览器访问 http://localhost:7860
```

## 📖 使用说明

### 上传图片识别
1. 切换到 "上传图片识别" 标签
2. 上传手写数字图片
3. 点击 "识别" 按钮
4. 查看 Top-3 预测和置信度

### 在线手写识别
1. 切换到 "在线手写识别" 标签
2. 在画板上绘制数字（0-9）
3. 点击 "识别手写数字" 按钮
4. 查看结果和概率分布
5. 点击 "清空画板" 重新绘制

## 📊 实验结果

| 实验 | 优化器 | Batch Size | 数据增强 | Early Stopping | 验证准确率 |
|------|--------|------------|----------|----------------|------------|
| Exp1 | SGD | 64 | ❌ | ❌ | 97.80% |
| Exp2 | Adam | 64 | ❌ | ❌ | 98.50% |
| Exp3 | Adam | 128 | ❌ | ✅ | 98.60% |
| **Exp4** | **Adam** | **64** | **✅** | **✅** | **99.35%** |

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.5.1+cu121
- **Web 框架**: Gradio 4.0+
- **图像处理**: Pillow 9.0+
- **科学计算**: NumPy 2.4.4, Pandas 3.0.2
- **可视化**: Matplotlib 3.10.9
- **部署平台**: HuggingFace Spaces (Free Tier)

## 📝 实验文档

- [实验报告](实验报告.md)
- [部署指南](部署指南.md)
- [实验完成总结](实验完成总结.md)

## 👤 学生信息

- **姓名**: 袁林阳
- **学号**: 112304260141
- **班级**: [请填写]
- **学校**: [请填写]

## 📄 许可证

MIT License © 2026 袁林阳

## 🙏 致谢

- MNIST 数据集
- PyTorch 团队
- Gradio 团队
- HuggingFace 云平台

---

**在线演示**: https://huggingface.co/spaces/yuan-linyang/mnist-digit-recognition  
**GitHub 仓库**: https://github.com/yuan-linyang/112304260141_yuan_linyang
