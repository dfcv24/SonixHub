#!/usr/bin/env python3
"""
SonixHub TTS API 客户端示例
演示如何在不同场景下使用语音合成API
"""

import requests
import json
import time
import os
import pygame
from io import BytesIO
from pathlib import Path


class SonixHubClient:
    """SonixHub TTS API客户端"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # 初始化pygame用于音频播放
        try:
            pygame.mixer.init()
            self.can_play_audio = True
        except:
            self.can_play_audio = False
            print("Warning: pygame未安装，无法播放音频")
    
    def check_status(self):
        """检查服务状态"""
        try:
            response = self.session.get(f"{self.base_url}/status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_languages(self):
        """获取支持的语言"""
        try:
            response = self.session.get(f"{self.base_url}/languages")
            if response.status_code == 200:
                return response.json()
            else:
                return {"languages": {}, "language_codes": []}
        except Exception as e:
            print(f"获取语言列表失败: {e}")
            return {"languages": {}, "language_codes": []}
    
    def synthesize_text(self, text, language="zh", save_path=None, play_audio=True, **kwargs):
        """
        基础文本合成
        
        Args:
            text: 要合成的文本
            language: 语言代码
            save_path: 保存路径
            play_audio: 是否播放音频
            **kwargs: 其他参数
        """
        try:
            payload = {
                "text": text,
                "text_language": language,
                "temperature": kwargs.get("temperature", 0.6),
                "speed": kwargs.get("speed", 1.0),
                "top_k": kwargs.get("top_k", 20),
                "top_p": kwargs.get("top_p", 0.6)
            }
            
            # 添加其他可选参数
            if "prompt_text" in kwargs:
                payload["prompt_text"] = kwargs["prompt_text"]
            if "prompt_language" in kwargs:
                payload["prompt_language"] = kwargs["prompt_language"]
            if "refer_wav_path" in kwargs:
                payload["refer_wav_path"] = kwargs["refer_wav_path"]
            
            print(f"正在合成: {text[:30]}...")
            response = self.session.post(f"{self.base_url}/tts", json=payload)
            
            if response.status_code == 200:
                # 保存音频文件
                if save_path:
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    print(f"音频已保存到: {save_path}")
                
                # 播放音频
                if play_audio and self.can_play_audio:
                    self.play_audio_from_bytes(response.content)
                
                return response.content
            else:
                print(f"合成失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"合成错误: {e}")
            return None
    
    def synthesize_with_reference(self, text, reference_audio_path, prompt_text, 
                                 language="zh", save_path=None, play_audio=True, **kwargs):
        """
        使用参考音频合成
        
        Args:
            text: 要合成的文本
            reference_audio_path: 参考音频路径
            prompt_text: 参考音频对应的文本
            language: 语言代码
            save_path: 保存路径
            play_audio: 是否播放音频
            **kwargs: 其他参数
        """
        try:
            if not os.path.exists(reference_audio_path):
                print(f"参考音频不存在: {reference_audio_path}")
                return None
            
            files = {"refer_audio": open(reference_audio_path, "rb")}
            data = {
                "text": text,
                "text_language": language,
                "prompt_text": prompt_text,
                "prompt_language": kwargs.get("prompt_language", language),
                "temperature": kwargs.get("temperature", 0.6),
                "speed": kwargs.get("speed", 1.0),
                "top_k": kwargs.get("top_k", 20),
                "top_p": kwargs.get("top_p", 0.6)
            }
            
            print(f"正在合成（使用参考音频）: {text[:30]}...")
            response = self.session.post(f"{self.base_url}/tts/upload", files=files, data=data)
            
            files["refer_audio"].close()
            
            if response.status_code == 200:
                # 保存音频文件
                if save_path:
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    print(f"音频已保存到: {save_path}")
                
                # 播放音频
                if play_audio and self.can_play_audio:
                    self.play_audio_from_bytes(response.content)
                
                return response.content
            else:
                print(f"合成失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"合成错误: {e}")
            return None
    
    def play_audio_from_bytes(self, audio_bytes):
        """从字节数据播放音频"""
        try:
            # 保存到临时文件
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            
            # 播放音频
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # 删除临时文件
            os.remove(temp_path)
            
        except Exception as e:
            print(f"播放音频失败: {e}")
    
    def batch_synthesize(self, texts, language="zh", output_dir="output", **kwargs):
        """批量合成语音"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, text in enumerate(texts):
            print(f"处理第 {i+1}/{len(texts)} 个文本...")
            output_path = os.path.join(output_dir, f"speech_{i+1:03d}.wav")
            
            audio_data = self.synthesize_text(
                text=text,
                language=language,
                save_path=output_path,
                play_audio=False,
                **kwargs
            )
            
            results.append({
                "index": i + 1,
                "text": text,
                "output_path": output_path,
                "success": audio_data is not None
            })
            
            time.sleep(0.5)  # 避免请求过于频繁
        
        return results


class VoiceAgent:
    """简单的语音Agent示例"""
    
    def __init__(self, tts_client):
        self.tts_client = tts_client
        self.conversation_history = []
    
    def speak(self, text, language="zh", **kwargs):
        """Agent说话"""
        print(f"Agent: {text}")
        
        # 记录对话历史
        self.conversation_history.append({
            "type": "agent",
            "text": text,
            "timestamp": time.time()
        })
        
        # 生成语音
        return self.tts_client.synthesize_text(
            text=text,
            language=language,
            play_audio=True,
            **kwargs
        )
    
    def process_user_input(self, user_input):
        """处理用户输入"""
        print(f"User: {user_input}")
        
        # 记录对话历史
        self.conversation_history.append({
            "type": "user",
            "text": user_input,
            "timestamp": time.time()
        })
        
        # 简单的回复逻辑
        if "你好" in user_input or "hello" in user_input.lower():
            response = "你好！我是SonixHub语音助手，很高兴见到你！"
        elif "再见" in user_input or "goodbye" in user_input.lower():
            response = "再见！期待下次与你对话！"
        elif "天气" in user_input:
            response = "今天天气不错，阳光明媚，适合外出活动。"
        elif "时间" in user_input:
            current_time = time.strftime("%H:%M:%S")
            response = f"现在时间是 {current_time}"
        else:
            response = f"我听到你说：{user_input}。这是一个很有趣的话题！"
        
        return self.speak(response)
    
    def chat_session(self):
        """开始聊天会话"""
        self.speak("你好！我是SonixHub语音助手，请问有什么可以帮助你的吗？")
        
        while True:
            try:
                user_input = input("\n请输入你的话（输入 'quit' 退出）: ")
                if user_input.lower() in ['quit', 'exit', '退出']:
                    self.speak("再见！感谢使用SonixHub语音助手！")
                    break
                
                if user_input.strip():
                    self.process_user_input(user_input)
                
            except KeyboardInterrupt:
                self.speak("再见！")
                break
    
    def save_conversation(self, filename="conversation.json"):
        """保存对话历史"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"对话历史已保存到: {filename}")


def main():
    """主函数 - 演示各种使用场景"""
    print("SonixHub TTS API 客户端示例")
    print("=" * 50)
    
    # 创建客户端
    client = SonixHubClient()
    
    # 检查服务状态
    print("1. 检查服务状态")
    status = client.check_status()
    print(f"服务状态: {status}")
    
    if status.get("status") != "running":
        print("错误: TTS服务未运行，请先启动服务")
        return
    
    # 获取支持的语言
    print("\n2. 获取支持的语言")
    languages = client.get_languages()
    print(f"支持的语言: {languages.get('language_codes', [])}")
    
    # 基础语音合成
    print("\n3. 基础语音合成")
    test_texts = [
        "你好，欢迎使用SonixHub语音合成服务！",
        "Hello, welcome to SonixHub TTS service!",
        "今天天气真不错，适合出门走走。"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n测试 {i+1}: {text}")
        client.synthesize_text(
            text=text,
            language="zh" if any(ord(c) > 127 for c in text) else "en",
            save_path=f"test_{i+1}.wav",
            play_audio=True,
            temperature=0.6,
            speed=1.0
        )
        time.sleep(1)
    
    # 批量合成
    print("\n4. 批量合成")
    batch_texts = [
        "这是第一段文本",
        "这是第二段文本",
        "这是第三段文本"
    ]
    
    results = client.batch_synthesize(
        texts=batch_texts,
        language="zh",
        output_dir="batch_output",
        temperature=0.7,
        speed=1.1
    )
    
    print("批量合成结果:")
    for result in results:
        print(f"  {result['index']}: {result['text'][:20]}... - {'成功' if result['success'] else '失败'}")
    
    # 交互式Agent演示
    print("\n5. 交互式Agent演示")
    choice = input("是否启动交互式Agent演示？(y/n): ").strip().lower()
    
    if choice == 'y':
        agent = VoiceAgent(client)
        agent.chat_session()
        agent.save_conversation()
    
    print("\n演示完成！")


if __name__ == "__main__":
    main()
