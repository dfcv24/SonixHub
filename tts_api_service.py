"""
TTS API服务
基于GPT-SoVITS的语音合成功能提供HTTP API接口
"""

import argparse
import os
import sys
import json
import traceback
from io import BytesIO
from typing import Optional

import torch
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# 导入我们的TTS核心模块
from tts_core import TTSCore, get_tts_instance


# 请求模型
class TTSRequest(BaseModel):
    text: str
    text_language: str = "zh"
    prompt_text: Optional[str] = None
    prompt_language: str = "zh"
    refer_wav_path: Optional[str] = None
    top_k: int = 20
    top_p: float = 0.6
    temperature: float = 0.6
    speed: float = 1.0
    how_to_cut: str = "凑四句一切"
    ref_free: bool = False
    if_freeze: bool = False
    sample_steps: int = 8
    if_sr: bool = False
    pause_second: float = 0.3


class ReferenceUpdateRequest(BaseModel):
    refer_wav_path: str
    prompt_text: str
    prompt_language: str = "zh"


class ModelChangeRequest(BaseModel):
    sovits_path: Optional[str] = None
    gpt_path: Optional[str] = None


# 全局变量
app = FastAPI(title="TTS API服务", description="基于GPT-SoVITS的语音合成API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 默认参考音频设置
default_refer_path = None
default_prompt_text = None
default_prompt_language = "zh"

# TTS实例
tts_instance = None


def init_api_service():
    """初始化API服务"""
    global default_refer_path, default_prompt_text, default_prompt_language, tts_instance
    
    # 初始化TTS实例
    try:
        tts_instance = get_tts_instance()
        print("TTS核心模块初始化成功")
    except Exception as e:
        print(f"TTS核心模块初始化失败: {e}")
        return
    
    # 设置默认参考音频（如果存在）
    possible_refer_paths = [
        "fanren150.wav",
        "GPT-SoVITS/fanren150.wav",
        "labixiaoxin.mp3"
    ]
    
    for path in possible_refer_paths:
        if os.path.exists(path):
            default_refer_path = path
            default_prompt_text = "一二三四五六七八九十"
            break
    
    print(f"API服务初始化完成")
    print(f"默认参考音频: {default_refer_path}")
    if tts_instance:
        print(f"设备: {tts_instance.device}")
        print(f"模型版本: {tts_instance.model_version}")
        print(f"支持的语言: {list(tts_instance.dict_language.keys())}")


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    init_api_service()


@app.get("/")
async def root():
    """根路径，返回API信息"""
    if tts_instance:
        supported_languages = list(tts_instance.dict_language.keys())
        model_version = tts_instance.model_version
        device = tts_instance.device
    else:
        supported_languages = []
        model_version = "未知"
        device = "未知"
    
    return {
        "message": "TTS API服务",
        "version": "1.0.0",
        "model_version": model_version,
        "device": device,
        "supported_languages": supported_languages,
        "endpoints": {
            "POST /tts": "语音合成",
            "POST /tts/upload": "上传参考音频并合成",
            "POST /change_refer": "更改默认参考音频",
            "POST /change_model": "更改模型",
            "GET /status": "获取服务状态"
        }
    }


@app.get("/status")
async def get_status():
    """获取服务状态"""
    if tts_instance:
        model_version = tts_instance.model_version
        device = tts_instance.device
    else:
        model_version = "未知"
        device = "未知"
    
    return {
        "status": "running" if tts_instance else "error",
        "model_version": model_version,
        "device": device,
        "default_refer_path": default_refer_path,
        "default_prompt_text": default_prompt_text,
        "default_prompt_language": default_prompt_language
    }


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """文本转语音"""
    global tts_instance
    
    if not tts_instance:
        raise HTTPException(status_code=500, detail="TTS服务未初始化")
    
    try:
        # 使用默认参考音频（如果未指定）
        refer_wav_path = request.refer_wav_path or default_refer_path
        prompt_text = request.prompt_text or default_prompt_text
        
        if not refer_wav_path:
            raise HTTPException(status_code=400, detail="未指定参考音频且没有默认参考音频")
        
        if not os.path.exists(refer_wav_path):
            raise HTTPException(status_code=400, detail=f"参考音频文件不存在: {refer_wav_path}")
        
        # 调用TTS核心的语音合成功能
        sr, audio_data = tts_instance.synthesize(
            text=request.text,
            text_language=request.text_language,
            refer_wav_path=refer_wav_path,
            prompt_text=prompt_text,
            prompt_language=request.prompt_language,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            speed=request.speed
        )
        
        # 将音频数据转换为WAV格式
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio_data, sr, format='WAV')
        audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
        )
        
    except Exception as e:
        print(f"TTS生成错误: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS生成失败: {str(e)}")


@app.post("/tts/upload")
async def text_to_speech_with_upload(
    text: str = Form(...),
    text_language: str = Form("zh"),
    prompt_text: str = Form(""),
    prompt_language: str = Form("zh"),
    refer_audio: UploadFile = File(...),
    top_k: int = Form(20),
    top_p: float = Form(0.6),
    temperature: float = Form(0.6),
    speed: float = Form(1.0),
    how_to_cut: str = Form("凑四句一切")
):
    """上传参考音频并进行语音合成"""
    global tts_instance
    
    if not tts_instance:
        raise HTTPException(status_code=500, detail="TTS服务未初始化")
    
    try:
        # 保存上传的参考音频
        temp_refer_path = f"temp_refer_{refer_audio.filename}"
        with open(temp_refer_path, "wb") as f:
            content = await refer_audio.read()
            f.write(content)
        
        try:
            # 调用语音合成
            sr, audio_data = tts_instance.synthesize(
                text=text,
                text_language=text_language,
                refer_wav_path=temp_refer_path,
                prompt_text=prompt_text,
                prompt_language=prompt_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                speed=speed
            )
            
            # 转换为WAV格式
            audio_buffer = BytesIO()
            sf.write(audio_buffer, audio_data, sr, format='WAV')
            audio_buffer.seek(0)
            
            return StreamingResponse(
                audio_buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
            )
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_refer_path):
                os.remove(temp_refer_path)
                
    except Exception as e:
        print(f"TTS生成错误: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS生成失败: {str(e)}")


@app.post("/change_refer")
async def change_reference(request: ReferenceUpdateRequest):
    """更改默认参考音频"""
    global default_refer_path, default_prompt_text, default_prompt_language
    
    try:
        if not os.path.exists(request.refer_wav_path):
            raise HTTPException(status_code=400, detail="参考音频文件不存在")
        
        default_refer_path = request.refer_wav_path
        default_prompt_text = request.prompt_text
        default_prompt_language = request.prompt_language
        
        return {
            "status": "success",
            "message": "默认参考音频已更新",
            "refer_wav_path": default_refer_path,
            "prompt_text": default_prompt_text,
            "prompt_language": default_prompt_language
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新参考音频失败: {str(e)}")


@app.post("/change_model")
async def change_model(request: ModelChangeRequest):
    """更改模型"""
    global tts_instance
    
    try:
        # 重新初始化TTS实例
        kwargs = {}
        if request.gpt_path:
            if not os.path.exists(request.gpt_path):
                raise HTTPException(status_code=400, detail="GPT模型文件不存在")
            kwargs["gpt_path"] = request.gpt_path
        
        if request.sovits_path:
            if not os.path.exists(request.sovits_path):
                raise HTTPException(status_code=400, detail="SoVITS模型文件不存在")
            kwargs["sovits_path"] = request.sovits_path
        
        # 创建新的TTS实例
        tts_instance = TTSCore(**kwargs)
        
        return {
            "status": "success",
            "message": "模型更新完成",
            "model_version": tts_instance.model_version,
            "device": tts_instance.device
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型更新失败: {str(e)}")


@app.get("/languages")
async def get_supported_languages():
    """获取支持的语言列表"""
    if tts_instance:
        return {
            "languages": tts_instance.dict_language,
            "language_codes": list(tts_instance.dict_language.keys())
        }
    else:
        return {
            "languages": {},
            "language_codes": []
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS API服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="开发模式，自动重载")
    
    args = parser.parse_args()
    
    print(f"TTS API服务启动中...")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"API文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "tts_api_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
