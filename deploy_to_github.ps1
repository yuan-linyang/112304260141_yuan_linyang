# GitHub 部署脚本
# 作者：袁林阳 (112304260141)
# 日期：2026-04-23

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MNIST 项目 - GitHub 仓库部署脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 进入项目目录
Set-Location "d:\神经网络\digit-recognizer"
Write-Host "[1/6] 进入项目目录..." -ForegroundColor Yellow
Write-Host "当前目录：$(Get-Location)" -ForegroundColor Gray
Write-Host ""

# 检查 Git 是否安装
Write-Host "[2/6] 检查 Git 安装..." -ForegroundColor Yellow
$gitVersion = git --version 2>$null
if ($gitVersion) {
    Write-Host "Git 已安装：$gitVersion" -ForegroundColor Green
} else {
    Write-Host "错误：Git 未安装！" -ForegroundColor Red
    Write-Host "请安装 Git: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# 初始化 Git 仓库（如果还没有）
Write-Host "[3/6] 初始化 Git 仓库..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    git init
    Write-Host "Git 仓库初始化完成" -ForegroundColor Green
} else {
    Write-Host "Git 仓库已存在" -ForegroundColor Yellow
}
Write-Host ""

# 配置 Git 用户信息（可选）
Write-Host "[可选] 配置 Git 用户信息..." -ForegroundColor Yellow
$configureGit = Read-Host "是否配置 Git 用户信息？(y/n)"
if ($configureGit -eq 'y' -or $configureGit -eq 'Y') {
    $userName = Read-Host "输入 GitHub 用户名"
    $userEmail = Read-Host "输入 GitHub 邮箱"
    git config user.name $userName
    git config user.email $userEmail
    Write-Host "Git 用户信息配置完成" -ForegroundColor Green
}
Write-Host ""

# 添加所有文件
Write-Host "[4/6] 添加所有文件..." -ForegroundColor Yellow
git add .
Write-Host "文件添加完成" -ForegroundColor Green
Write-Host ""

# 查看将要提交的文件
Write-Host "将要提交的文件列表:" -ForegroundColor Cyan
git status --short
Write-Host ""

# 确认提交
$confirm = Read-Host "确认提交并推送到 GitHub？(y/n)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "操作已取消" -ForegroundColor Yellow
    exit 0
}
Write-Host ""

# 提交
Write-Host "[5/6] 提交更改..." -ForegroundColor Yellow
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
- Exp4: Adam + Data Augmentation + ES (99.35% val acc)

Author: Yuan Linyang (112304260141)
Date: $(Get-Date -Format 'yyyy-MM-dd')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "提交成功" -ForegroundColor Green
} else {
    Write-Host "没有需要提交的更改" -ForegroundColor Yellow
}
Write-Host ""

# 关联远程仓库
Write-Host "[6/6] 关联远程仓库..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin https://github.com/yuan-linyang/112304260141_yuan_linyang.git
Write-Host "远程仓库关联完成" -ForegroundColor Green
Write-Host ""

# 推送到 GitHub
Write-Host "准备推送到 GitHub..." -ForegroundColor Yellow
Write-Host "⚠️  警告：这将覆盖远程仓库的所有内容！" -ForegroundColor Red
Write-Host ""

$confirmPush = Read-Host "确认强制推送到 GitHub？(y/n)"
if ($confirmPush -ne 'y' -and $confirmPush -ne 'Y') {
    Write-Host "操作已取消" -ForegroundColor Yellow
    exit 0
}
Write-Host ""

# 设置主分支
git branch -M main

# 推送
Write-Host "正在推送到 GitHub... (这可能需要几分钟)" -ForegroundColor Yellow
Write-Host ""

# 使用 --force-with-lease 更安全
git push --force-with-lease origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "部署完成！✅" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "访问仓库：" -NoNewline
    Write-Host "https://github.com/yuan-linyang/112304260141_yuan_linyang" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "下一步操作:" -ForegroundColor Cyan
    Write-Host "1. 访问仓库确认文件已上传" -ForegroundColor White
    Write-Host "2. 检查 README.md 是否正确显示" -ForegroundColor White
    Write-Host "3. 部署到 HuggingFace Spaces (参考：部署指南.md)" -ForegroundColor White
    Write-Host "4. 提交 Kaggle 竞赛 (sample_submission.csv)" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "推送失败！❌" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "可能的原因:" -ForegroundColor Yellow
    Write-Host "1. 网络连接问题" -ForegroundColor White
    Write-Host "2. GitHub 账号权限问题" -ForegroundColor White
    Write-Host "3. 仓库不存在或已被删除" -ForegroundColor White
    Write-Host ""
    Write-Host "解决方案:" -ForegroundColor Yellow
    Write-Host "1. 检查网络连接" -ForegroundColor White
    Write-Host "2. 确认已登录 GitHub 账号" -ForegroundColor White
    Write-Host "3. 手动创建仓库后重试" -ForegroundColor White
    Write-Host ""
}

# 恢复目录
Set-Location "d:\神经网络"
