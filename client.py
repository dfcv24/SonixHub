import requests
import json
import os

class SonixHubClient:
    """SonixHub TTS API客户端"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """检查服务健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def synthesize_speech(self, text, engine="gtts", language="zh"):
        """
        合成语音
        
        Args:
            text (str): 要合成的文本
            engine (str): 引擎类型，"gtts" 或 "pyttsx3"
            language (str): 语言代码，如 "zh", "en"
        
        Returns:
            dict: 包含文件ID和下载URL的响应
        """
        try:
            payload = {
                "text": text,
                "engine": engine,
                "language": language
            }
            
            response = requests.post(
                f"{self.base_url}/tts/synthesize",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def download_audio(self, file_id, output_path=None):
        """
        下载音频文件
        
        Args:
            file_id (str): 文件ID
            output_path (str): 输出路径，如果为None则使用默认路径
        
        Returns:
            str: 保存的文件路径
        """
        try:
            response = requests.get(f"{self.base_url}/tts/download/{file_id}")
            
            if response.status_code == 200:
                if not output_path:
                    # 从响应头获取文件名
                    content_disposition = response.headers.get('content-disposition', '')
                    if 'filename=' in content_disposition:
                        filename = content_disposition.split('filename=')[1].strip('"')
                    else:
                        filename = f"speech_{file_id}.mp3"
                    
                    output_path = os.path.join(os.getcwd(), filename)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                return output_path
            else:
                return None
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    def stream_speech(self, text, language="zh", output_path=None):
        """
        流式语音合成
        
        Args:
            text (str): 要合成的文本
            language (str): 语言代码
            output_path (str): 输出路径
        
        Returns:
            str: 保存的文件路径
        """
        try:
            payload = {
                "text": text,
                "language": language
            }
            
            response = requests.post(
                f"{self.base_url}/tts/stream",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                if not output_path:
                    output_path = os.path.join(os.getcwd(), "speech_stream.mp3")
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                return output_path
            else:
                return None
        except Exception as e:
            print(f"Stream error: {e}")
            return None
    
    def get_available_voices(self):
        """获取可用的语音引擎和语言"""
        try:
            response = requests.get(f"{self.base_url}/tts/voices")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def speak(self, text, engine="gtts", language="zh", save_path=None):
        """
        便捷方法：合成并下载语音
        
        Args:
            text (str): 要合成的文本
            engine (str): 引擎类型
            language (str): 语言代码
            save_path (str): 保存路径
        
        Returns:
            str: 保存的文件路径
        """
        # 合成语音
        result = self.synthesize_speech(text, engine, language)
        
        if "error" in result:
            print(f"Synthesis error: {result['error']}")
            return None
        
        # 下载音频文件
        file_id = result.get("file_id")
        if file_id:
            return self.download_audio(file_id, save_path)
        
        return None

# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = SonixHubClient()
    
    # 检查服务状态
    print("Health Check:", client.health_check())
    
    # 获取可用语音
    print("Available Voices:", client.get_available_voices())
    
    # 合成中文语音
    print("Synthesizing Chinese speech...")
    audio_path = client.speak("你好，我是SonixHub语音合成服务！", engine="gtts", language="zh")
    if audio_path:
        print(f"Audio saved to: {audio_path}")
    
    # 合成英文语音
    print("Synthesizing English speech...")
    audio_path = client.speak("Hello, I am SonixHub TTS service!", engine="gtts", language="en")
    if audio_path:
        print(f"Audio saved to: {audio_path}")
    
    # 使用流式API
    print("Using stream API...")
    stream_path = client.stream_speech("这是流式语音合成测试", language="zh")
    if stream_path:
        print(f"Stream audio saved to: {stream_path}")
