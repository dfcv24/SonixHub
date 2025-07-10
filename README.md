# SonixHub

🎙️ **SonixHub** 是一个基于GPT-SoVITS的语音合成API服务，为外部agent项目提供高质量的语音合成能力。

## ✨ 特性

- 🚀 **高质量语音合成** - 基于GPT-SoVITS技术，支持多语言语音合成
- 🌍 **多语言支持** - 支持中文、英文、日文、粤语、韩语等多种语言
- 🎯 **RESTful API** - 提供简单易用的HTTP API接口
- 🔧 **灵活配置** - 支持自定义参考音频、语音参数调节
- 📁 **文件上传** - 支持实时上传参考音频文件
- 🎨 **参数调节** - 支持语速、语调、温度等参数精细调节

## 🛠️ 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-repo/SonixHub.git
cd SonixHub

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 准备模型文件

1. 下载GPT-SoVITS预训练模型
2. 将模型文件放在相应目录下
3. 准备参考音频文件（用于声音克隆）

### 启动服务

```bash
# 方式1：使用启动脚本
./start.sh

# 方式2：直接运行
python tts_api_service.py --host 0.0.0.0 --port 8000

# 方式3：开发模式（自动重载）
python tts_api_service.py --host 0.0.0.0 --port 8000 --reload
```

服务启动后，可以通过以下地址访问：
- API服务: http://localhost:8000
- API文档: http://localhost:8000/docs
- 交互式API: http://localhost:8000/redoc

## 📚 API文档

### 1. 基础信息接口

#### 获取服务信息
```bash
GET /
```

**响应示例：**
```json
{
  "message": "TTS API服务",
  "version": "1.0.0",
  "model_version": "v2",
  "device": "cuda",
  "supported_languages": ["中文", "英文", "日文", "中英混合"],
  "endpoints": {
    "POST /tts": "语音合成",
    "POST /tts/upload": "上传参考音频并合成",
    "POST /change_refer": "更改默认参考音频",
    "POST /change_model": "更改模型",
    "GET /status": "获取服务状态"
  }
}
```

#### 获取服务状态
```bash
GET /status
```

**响应示例：**
```json
{
  "status": "running",
  "model_version": "v2",
  "device": "cuda",
  "default_refer_path": "fanren150.wav",
  "default_prompt_text": "一二三四五六七八九十",
  "default_prompt_language": "zh"
}
```

### 2. 语音合成接口

#### 基础语音合成
```bash
POST /tts
Content-Type: application/json

{
  "text": "你好，我是SonixHub语音合成服务！",
  "text_language": "zh",
  "prompt_text": "一二三四五六七八九十",
  "prompt_language": "zh",
  "refer_wav_path": "fanren150.wav",
  "top_k": 20,
  "top_p": 0.6,
  "temperature": 0.6,
  "speed": 1.0
}
```

**参数说明：**
- `text` (必填): 要合成的文本
- `text_language` (可选): 文本语言，默认"zh"
- `prompt_text` (可选): 参考音频对应的文本
- `prompt_language` (可选): 参考音频语言，默认"zh"
- `refer_wav_path` (可选): 参考音频文件路径
- `top_k` (可选): 采样参数，默认20
- `top_p` (可选): 采样参数，默认0.6
- `temperature` (可选): 温度参数，默认0.6
- `speed` (可选): 语速，默认1.0

**响应：** 返回WAV格式的音频文件

#### 上传参考音频合成
```bash
POST /tts/upload
Content-Type: multipart/form-data

text=你好世界
text_language=zh
prompt_text=测试音频
prompt_language=zh
refer_audio=<音频文件>
top_k=20
top_p=0.6
temperature=0.6
speed=1.0
```

**响应：** 返回WAV格式的音频文件

### 3. 配置管理接口

#### 更改默认参考音频
```bash
POST /change_refer
Content-Type: application/json

{
  "refer_wav_path": "new_reference.wav",
  "prompt_text": "新的参考文本",
  "prompt_language": "zh"
}
```

#### 更改模型
```bash
POST /change_model
Content-Type: application/json

{
  "gpt_path": "path/to/gpt/model.ckpt",
  "sovits_path": "path/to/sovits/model.pth"
}
```

#### 获取支持的语言
```bash
GET /languages
```

## 💡 使用示例

### Python客户端示例

```python
import requests
import json

# 基础语音合成
def synthesize_speech(text, output_path="output.wav"):
    url = "http://localhost:8000/tts"
    payload = {
        "text": text,
        "text_language": "zh",
        "temperature": 0.6,
        "speed": 1.0
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"音频已保存到: {output_path}")
    else:
        print(f"合成失败: {response.status_code}")

# 使用示例
synthesize_speech("你好，欢迎使用SonixHub语音合成服务！")
```

### 使用curl命令

```bash
# 基础语音合成
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一个测试",
    "text_language": "zh",
    "temperature": 0.6,
    "speed": 1.0
  }' \
  --output output.wav

# 上传参考音频合成
curl -X POST "http://localhost:8000/tts/upload" \
  -F "text=你好世界" \
  -F "text_language=zh" \
  -F "prompt_text=测试音频" \
  -F "prompt_language=zh" \
  -F "refer_audio=@reference.wav" \
  --output output.wav

# 获取服务状态
curl -X GET "http://localhost:8000/status"
```

### JavaScript客户端示例

```javascript
// 语音合成
async function synthesizeSpeech(text) {
    const response = await fetch('http://localhost:8000/tts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            text_language: 'zh',
            temperature: 0.6,
            speed: 1.0
        })
    });
    
    if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // 创建音频元素播放
        const audio = new Audio(url);
        audio.play();
        
        return url;
    } else {
        console.error('合成失败:', response.status);
    }
}

// 使用示例
synthesizeSpeech('你好，欢迎使用SonixHub！');
```

## 🔧 高级配置

### 语言支持

| 语言 | 代码 | 说明 |
|-----|------|------|
| 中文 | zh | 全部按中文识别 |
| 英文 | en | 全部按英文识别 |
| 日文 | ja | 全部按日文识别 |
| 粤语 | yue | 全部按粤语识别 |
| 韩文 | ko | 全部按韩文识别 |
| 中英混合 | zh | 按中英混合识别 |
| 日英混合 | ja | 按日英混合识别 |
| 多语种混合 | auto | 自动识别语种 |

### 参数调节指南

| 参数 | 范围 | 说明 |
|-----|------|------|
| `top_k` | 1-100 | 采样时考虑的候选词数量，值越大多样性越高 |
| `top_p` | 0.0-1.0 | 核采样参数，控制采样的随机性 |
| `temperature` | 0.0-2.0 | 温度参数，控制输出的随机性 |
| `speed` | 0.5-2.0 | 语速倍率，1.0为正常语速 |

### 模型文件结构

```
SonixHub/
├── GPT-SoVITS/
│   ├── pretrained_models/
│   │   ├── chinese-hubert-base/
│   │   ├── chinese-roberta-wwm-ext-large/
│   │   └── s2G_v4.pth
│   └── ...
├── weights/
│   ├── gpt/
│   │   └── your_gpt_model.ckpt
│   └── sovits/
│       └── your_sovits_model.pth
└── reference_audio/
    └── your_reference.wav
```

## 🎯 Agent集成示例

### 简单Agent集成

```python
class VoiceAgent:
    def __init__(self, tts_url="http://localhost:8000"):
        self.tts_url = tts_url
    
    def speak(self, text, language="zh"):
        """让Agent说话"""
        response = requests.post(
            f"{self.tts_url}/tts",
            json={
                "text": text,
                "text_language": language,
                "temperature": 0.7,
                "speed": 1.1
            }
        )
        
        if response.status_code == 200:
            # 保存音频文件
            with open("agent_speech.wav", "wb") as f:
                f.write(response.content)
            
            # 播放音频（需要安装pygame或其他音频库）
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load("agent_speech.wav")
            pygame.mixer.music.play()
            
            return True
        return False
    
    def chat_with_voice(self, user_input):
        """带语音的聊天功能"""
        # 这里可以集成ChatGPT等AI模型
        ai_response = f"你说的是：{user_input}，我理解了。"
        
        # 生成语音回复
        if self.speak(ai_response):
            print(f"AI回复: {ai_response}")
            return ai_response
        else:
            print("语音生成失败")
            return None

# 使用示例
agent = VoiceAgent()
agent.chat_with_voice("你好，今天天气怎么样？")
```

## 🚀 部署指南

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 复制项目文件
COPY . .

# 安装依赖
RUN pip install -r requirements.txt

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "tts_api_service.py", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 构建镜像
docker build -t sonixhub:latest .

# 运行容器
docker run -p 8000:8000 sonixhub:latest
```

### 生产环境部署

```bash
# 使用Gunicorn部署
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker tts_api_service:app --bind 0.0.0.0:8000

# 使用Nginx反向代理
# /etc/nginx/sites-available/sonixhub
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 🐛 常见问题

### Q: 模型加载失败怎么办？
A: 请检查模型文件路径是否正确，确保已下载所有必要的预训练模型。

### Q: 内存不足怎么处理？
A: 可以设置 `is_half=True` 使用半精度模式，或者使用CPU模式。

### Q: 语音质量不佳怎么办？
A: 尝试调整 `temperature`、`top_k`、`top_p` 参数，或者使用更好的参考音频。

### Q: 如何添加新的语言支持？
A: 需要相应的语言模型和训练数据，具体请参考GPT-SoVITS的官方文档。

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📞 联系我们

- 项目主页: https://github.com/your-repo/SonixHub
- 问题反馈: https://github.com/your-repo/SonixHub/issues
- 邮箱: your-email@example.com

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！
