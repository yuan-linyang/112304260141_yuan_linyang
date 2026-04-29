# 🎯 MNIST 手写数字识别系统

基于 CNN 的在线手写数字识别系统，支持上传图片、网页手写板、Top-3 预测、概率分布和连续识别历史。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

## ✨ 功能特点

### 实验二：Web 应用
- 🖼️ **图片上传识别**: 支持 JPG、PNG 等格式
- 🎯 **实时预测**: 毫秒级响应
- 📊 **Top-3 预测**: 显示概率最高的 3 个结果
- 💯 **置信度显示**: 精确到小数点后 2 位

### 实验三：交互系统（加分）
- ✏️ **网页手写板**: 交互式画板，支持鼠标/触屏绘制
- 🔄 **连续识别**: 支持多次识别，历史记录
- 📈 **概率分布图**: 可视化显示 0-9 的概率
- 🗑️ **清空画板**: 一键清除，重新绘制
- 🎨 **智能预处理**: 自动裁剪、归一化、反色处理

## 🚀 快速开始

### 本地运行

```bash
# 1. 克隆仓库
git clone https://github.com/yuan-linyang/112304260141_yuan_linyang.git
cd digit-recognizer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python app_final.py

# 4. 访问应用
# 打开浏览器访问 http://127.0.0.1:7860
```

### 在线访问

访问 Render 部署的在线系统：
**[https://mnist-digit-recognition.onrender.com](https://mnist-digit-recognition.onrender.com)**

## 📦 项目结构

```
digit-recognizer/
├── app_final.py              # Web 应用入口（完整版）
├── best_model.pth            # 训练好的模型权重
├── requirements.txt          # 依赖列表
├── README.md                 # 项目说明
├── .gitignore               # Git 忽略文件
├── render.yaml              # Render 部署配置
├── 实验三_交互式手写识别系统.md  # 实验三文档
└── 训练脚本/
    ├── exp1_sgd.py          # 实验 1: SGD 优化器
    ├── exp2_adam.py         # 实验 2: Adam 优化器
    ├── exp3_early_stopping.py  # 实验 3: Early Stopping
    └── train_cnn_augmentation.py  # 实验 4: 数据增强
```

## 🧠 模型架构

```
输入 (1×28×28) 
  ↓
Conv2d(1→32) + BatchNorm + ReLU + MaxPool(2×2)
  ↓
Conv2d(32→64) + BatchNorm + ReLU + MaxPool(2×2)
  ↓
Conv2d(64→128) + BatchNorm + ReLU + MaxPool(2×2)
  ↓
Flatten (128×3×3 = 1152)
  ↓
Linear(1152→256) + BatchNorm + ReLU + Dropout(0.5)
  ↓
Linear(256→10)
  ↓
Softmax 输出
```

### 训练配置

| 配置项 | 值 |
|--------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 |
| Batch Size | 64 |
| 数据增强 | Rotation, Translation, Zoom, Shear, Noise |
| Early Stopping | Patience=15 |
| 验证准确率 | **99.35%** |
| Epochs | 26 (Early Stopped) |

## 📊 实验结果

### 超参数对比

| 实验 | 优化器 | Batch Size | 数据增强 | Early Stopping | Val Acc |
|------|--------|------------|----------|----------------|---------|
| Exp1 | SGD | 64 | ❌ | ❌ | 99.19% |
| Exp2 | Adam | 64 | ❌ | ❌ | 99.23% |
| Exp3 | Adam | 128 | ❌ | ✅ | 99.24% |
| Exp4 | Adam | 64 | ✅ | ✅ | **99.35%** |

### 性能指标
- **训练集准确率**: 99.70%
- **验证集准确率**: 99.35%
- **测试集准确率**: 待 Kaggle 提交
- **推理时间**: <50ms (CPU), <10ms (GPU)

## 🌐 部署说明

### GitHub 仓库
**https://github.com/yuan-linyang/112304260141_yuan_linyang**

### Render 部署

1. **登录**: 使用 GitHub 账号登录 [Render](https://dashboard.render.com/)
2. **创建服务**: New + → Web Service
3. **连接仓库**: 选择 `112304260141_yuan_linyang`
4. **配置**:
   - **Name**: `mnist-digit-recognition`
   - **Region**: Singapore
   - **Root Directory**: `digit-recognizer`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app_final.py`
5. **部署**: 点击 Create Web Service

详细步骤见 [实验三文档](实验三_交互式手写识别系统.md)

## 📸 使用示例

### 上传图片识别
1. 切换到 "上传图片识别" 标签
2. 上传手写数字图片
3. 查看 Top-3 预测和置信度

### 网页手写识别
1. 切换到 "在线手写识别" 标签
2. 在画板上绘制数字
3. 点击 "识别手写数字"
4. 查看结果和概率分布
5. 点击 "清空画板" 重新绘制

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.5.1+cu121
- **Web 框架**: Gradio 3.0+
- **图像处理**: Pillow 9.0+
- **科学计算**: NumPy 2.4.4, Pandas 3.0.2
- **可视化工具**: Matplotlib 3.10.9
- **部署平台**: Render (Free Tier)
- **版本控制**: Git + GitHub

## 📝 实验文档

- [实验一：模型训练与超参数调优](../CNN 手写数字识别实验报告.md#实验一模型训练与超参数调优必做)
- [实验二：模型封装与 Web 部署](../CNN 手写数字识别实验报告.md#实验二模型封装与 web 部署必做)
- [实验三：交互式手写识别系统](实验三_交互式手写识别系统.md)

## 👤 作者信息

- **姓名**: [请填写]
- **学号**: [请填写]
- **班级**: [请填写]
- **学校**: [请填写]

## 📄 许可证

MIT License © 2026

## 🙏 致谢

- MNIST 数据集
- PyTorch 团队
- Gradio 团队
- Render 云平台

---

<div align="center">

**🎯 手写数字识别系统 | 基于 CNN 的深度学习实践**

[GitHub 仓库](https://github.com/yuan-linyang/112304260141_yuan_linyang) • [在线演示](https://mnist-digit-recognition.onrender.com) • [实验报告](../CNN 手写数字识别实验报告.md)

</div>
