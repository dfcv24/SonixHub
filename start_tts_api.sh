#!/bin/bash

# TTS API服务启动脚本

echo "正在启动TTS API服务..."

# 检查Python版本
python_version=$(python3 --version 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python3"
    exit 1
fi

echo "Python版本: $python_version"

# 检查是否存在GPT-SoVITS目录
if [ ! -d "GPT-SoVITS" ]; then
    echo "错误: GPT-SoVITS目录不存在"
    echo "请确保已经正确克隆了GPT-SoVITS项目"
    exit 1
fi

# 安装依赖
echo "安装API服务依赖..."
pip3 install -r requirements_api.txt

# 检查是否有GPU可用
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU"
    gpu_available=true
else
    echo "未检测到NVIDIA GPU，将使用CPU"
    gpu_available=false
fi

# 启动服务
echo "启动TTS API服务..."
python3 tts_api_service.py --host 0.0.0.0 --port 8000

echo "服务已启动，访问地址: http://localhost:8000"
echo "API文档地址: http://localhost:8000/docs"
