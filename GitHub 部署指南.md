# 🚀 GitHub 仓库部署指南

## 📋 部署目标

将 `d:\神经网络\digit-recognizer` 目录下的完整项目部署到 GitHub 仓库：
- **仓库地址**: https://github.com/yuan-linyang/112304260141_yuan_linyang
- **操作**: 删除旧内容，上传新项目

---

## ⚠️ 重要提示

### 部署前备份
```bash
# 备份当前仓库内容（可选）
git clone https://github.com/yuan-linyang/112304260141_yuan_linyang.git backup_repo
```

### 确认删除
- ⚠️ 此操作会**永久删除**仓库中所有现有文件
- ⚠️ 请确保已备份重要文件
- ✅ 确认后再执行后续步骤

---

## 方法一：使用 Git 命令行（推荐）

### Step 1: 初始化 Git 仓库

```powershell
# 进入项目目录
cd d:\神经网络\digit-recognizer

# 初始化 Git 仓库（如果还没有）
git init

# 添加所有文件
git add .

# 提交
git commit -m "Complete MNIST digit recognition project

Project includes:
- 4 experiment scripts (SGD, Adam, Early Stopping, Data Augmentation)
- CNN model with 99.35% validation accuracy
- Gradio web application with image upload and sketchpad
- Complete experiment reports and documentation

Experiments:
- Exp1: SGD optimizer (97.80% val acc)
- Exp2: Adam optimizer (98.50% val acc)
- Exp3: Adam + Early Stopping (98.60% val acc)
- Exp4: Adam + Aug + ES (99.35% val acc)

Author: Yuan Linyang (112304260141)
Date: 2026-04-23"
```

### Step 2: 关联远程仓库

```powershell
# 添加远程仓库（如果已关联，先移除旧的）
git remote remove origin 2>$null

# 添加新的远程仓库
git remote add origin https://github.com/yuan-linyang/112304260141_yuan_linyang.git

# 查看远程仓库
git remote -v
```

### Step 3: 推送到 GitHub

```powershell
# 强制推送（会覆盖远程仓库所有内容）
git branch -M main

# 强制推送（⚠️ 会删除远程仓库所有旧文件）
git push -f origin main

# 或者使用 --force-with-lease（更安全）
git push --force-with-lease origin main
```

### Step 4: 验证推送

访问仓库确认文件已上传：
```
https://github.com/yuan-linyang/112304260141_yuan_linyang
```

---

## 方法二：使用 GitHub Desktop（适合新手）

### Step 1: 安装 GitHub Desktop
- 下载地址：https://desktop.github.com
- 安装并登录 GitHub 账号

### Step 2: 添加本地仓库
1. 打开 GitHub Desktop
2. File → Add Local Repository
3. 选择目录：`d:\神经网络\digit-recognizer`
4. 如果提示不是 Git 仓库，点击 "Create a repository"

### Step 3: 关联远程仓库
1. Repository → Repository Settings
2. 在 "Primary remote repository" 点击 "Change"
3. 选择：`yuan-linyang/112304260141_yuan_linyang`
4. 点击 "Save"

### Step 4: 推送更改
1. 在 Changes 标签页查看所有更改
2. 填写提交信息：
   - Summary: `Complete MNIST project with experiments and web app`
   - Description: 详细描述（可选）
3. 点击 "Commit to main"
4. 点击 "Push origin" 推送到 GitHub

---

## 方法三：使用 GitHub 网页（最简单但繁琐）

### Step 1: 删除旧文件
1. 访问 https://github.com/yuan-linyang/112304260141_yuan_linyang
2. 逐个点击文件 → 点击右上角垃圾桶图标 → Commit changes
3. 或者使用以下方法批量删除：

```powershell
# 使用 Git 命令行批量删除
git clone https://github.com/yuan-linyang/112304260141_yuan_linyang.git temp_repo
cd temp_repo
git rm -rf .
git commit -m "Remove all old files"
git push
cd ..
rmdir /s temp_repo
```

### Step 2: 上传新文件
1. 点击 "Add file" → "Upload files"
2. 拖拽所有文件到上传区域
3. 填写提交信息
4. 点击 "Commit changes"

⚠️ **注意**: 此方法不适合大文件（如 .pth 模型文件），建议使用 Git 命令行。

---

## 📁 推荐上传的文件清单

### ✅ 核心文件（必需）
```
app.py                      # Web 应用主程序
best_model.pth              # 最佳模型权重（1.57MB）
requirements.txt            # Python 依赖包
README.md                   # 完整实验报告
.gitignore                  # Git 忽略文件
```

### ✅ 实验一文件
```
exp1_sgd.py                 # SGD 优化器训练
exp2_adam.py                # Adam 优化器训练
exp3_early_stopping.py      # Early Stopping 训练
exp4_data_augmentation.py   # 数据增强训练
plot_comparison.py          # 对比图表绘制
model_exp1_sgd.pth          # Exp1 模型
model_exp2_adam.pth         # Exp2 模型
exp1_results.npy            # Exp1 结果
exp2_results.npy            # Exp2 结果
exp3_results.npy            # Exp3 结果
exp4_results.npy            # Exp4 结果
sample_submission.csv       # Kaggle 提交文件
loss_curve_comparison.png   # Loss 曲线对比图
```

### ✅ 文档文件
```
实验报告.md                 # 详细实验报告
实验完成总结.md             # 项目总结
部署指南.md                 # HuggingFace 部署教程
部署清单.md                 # 部署检查清单
快速部署.md                 # 快速部署脚本
文件索引.md                 # 文件导航索引
```

### ✅ 可视化图表
```
loss_curve_comparison.png              # 4 组实验对比
data_augmentation_examples.png         # 数据增强示例
training_curves_with_augmentation.png  # 数据增强训练曲线
```

### ✅ 数据集
```
train.csv                     # 训练集（42,000 样本）
test.csv                      # 测试集（28,000 样本）
```

---

## ⚠️ 不需要上传的文件

### 大文件（>50MB）
```
best_cnn_model.pth            # 已有 best_model.pth
best_cnn_model_100.pth        # 重复文件
model_exp3.pth                # 重复文件
```

### 临时文件
```
__pycache__/                  # Python 缓存
*.pyc                         # 编译的 Python 文件
*.log                         # 日志文件
```

### 重复文档
```
README_Space.md               # 与 README.md 重复
实验一完成.md                 # 内容与实验报告重复
```

---

## 🔧 常见问题

### Q1: 推送失败 "Permission denied"
**A**: 
```powershell
# 检查 SSH key 或 HTTPS 凭证
git config --global credential.helper wincred

# 重新输入 GitHub 账号密码
git push -f origin main
```

### Q2: 文件太大无法推送
**A**:
```powershell
# 检查大文件
git ls-files | xargs ls -lh | sort -k5 -rn | head -10

# 如果 best_model.pth 太大，使用 Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add .
git commit -m "Add Git LFS tracking for model files"
git push -f origin main
```

### Q3: 如何撤销强制推送
**A**:
```powershell
# 查看推送前的 commit
git reflog

# 重置到推送前的状态
git reset --hard HEAD@{1}

# 恢复远程仓库
git push -f origin main
```

---

## ✅ 部署后检查

### 1. 验证文件完整性
访问仓库确认以下文件存在：
- [ ] README.md（包含完整实验报告）
- [ ] app.py
- [ ] best_model.pth
- [ ] requirements.txt
- [ ] 所有实验脚本（exp1-4.py）
- [ ] 所有文档（*.md）

### 2. 验证 README 显示
- [ ] README.md 内容正确显示
- [ ] 实验报告格式正确
- [ ] 图片正常加载（loss_curve_comparison.png）

### 3. 测试克隆
```powershell
# 在新目录测试克隆
cd d:\神经网络
git clone https://github.com/yuan-linyang/112304260141_yuan_linyang.git test_clone
cd test_clone

# 检查文件
ls

# 清理测试
cd ..
rmdir /s test_clone
```

---

## 📊 预计上传时间和大小

### 文件大小统计
| 类型 | 数量 | 总大小 |
|------|------|--------|
| Python 脚本 | ~10 个 | ~100 KB |
| 模型文件 | ~3 个 | ~5 MB |
| 文档 | ~9 个 | ~50 KB |
| 图片 | ~5 个 | ~2 MB |
| 数据集 | 2 个 | ~60 MB |
| **总计** | **~29 个** | **~67 MB** |

### 上传时间估算
- **10 Mbps 网络**: ~1 分钟
- **50 Mbps 网络**: ~15 秒
- **100 Mbps 网络**: ~8 秒

---

## 🎯 快速部署脚本

创建一个 PowerShell 脚本 `deploy_to_github.ps1`：

```powershell
# deploy_to_github.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub 仓库部署脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 进入项目目录
Set-Location "d:\神经网络\digit-recognizer"
Write-Host "[1/5] 进入项目目录..." -ForegroundColor Yellow

# 初始化 Git（如果还没有）
if (-not (Test-Path ".git")) {
    git init
    Write-Host "Git 仓库初始化完成" -ForegroundColor Green
} else {
    Write-Host "Git 仓库已存在" -ForegroundColor Yellow
}

# 添加所有文件
Write-Host "[2/5] 添加所有文件..." -ForegroundColor Yellow
git add .

# 提交
Write-Host "[3/5] 提交更改..." -ForegroundColor Yellow
git commit -m "Complete MNIST digit recognition project

Project includes:
- 4 experiment scripts with 99.35% validation accuracy
- Gradio web application
- Complete experiment reports and documentation

Author: Yuan Linyang (112304260141)
Date: $(Get-Date -Format 'yyyy-MM-dd')"

# 关联远程仓库
Write-Host "[4/5] 关联远程仓库..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin https://github.com/yuan-linyang/112304260141_yuan_linyang.git

# 推送到 GitHub
Write-Host "[5/5] 推送到 GitHub (这可能需要几分钟)..." -ForegroundColor Yellow
git branch -M main

# 使用 --force-with-lease 更安全
git push --force-with-lease origin main

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "部署完成！✅" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "访问仓库：" -NoNewline
Write-Host "https://github.com/yuan-linyang/112304260141_yuan_linyang" -ForegroundColor Cyan
Write-Host ""
```

### 运行部署脚本

```powershell
# 在 PowerShell 中运行
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\deploy_to_github.ps1
```

---

## 📞 获取帮助

如遇到问题：
1. 检查 Git 版本：`git --version`
2. 检查网络连接
3. 确认 GitHub 账号有写入权限
4. 查看 Git 日志：`git log --oneline`

---

**祝你部署顺利！🎉**

**最后更新**: 2026 年 4 月 23 日  
**作者**: 袁林阳 (112304260141)
