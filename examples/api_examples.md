# SonixHub TTS API 调用示例

## Python 示例

### 基础调用
```python
import requests
import json

def synthesize_speech(text, language="zh", output_file="output.wav"):
    """基础语音合成"""
    url = "http://localhost:8000/tts"
    payload = {
        "text": text,
        "text_language": language,
        "temperature": 0.6,
        "speed": 1.0,
        "top_k": 20,
        "top_p": 0.6
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"语音已保存到: {output_file}")
        return True
    else:
        print(f"合成失败: {response.status_code}")
        return False

# 使用示例
synthesize_speech("你好，欢迎使用SonixHub语音合成服务！")
```

### 使用参考音频
```python
import requests

def synthesize_with_reference(text, reference_audio_path, prompt_text, output_file="output.wav"):
    """使用参考音频合成"""
    url = "http://localhost:8000/tts/upload"
    
    files = {"refer_audio": open(reference_audio_path, "rb")}
    data = {
        "text": text,
        "text_language": "zh",
        "prompt_text": prompt_text,
        "prompt_language": "zh",
        "temperature": 0.6,
        "speed": 1.0
    }
    
    response = requests.post(url, files=files, data=data)
    files["refer_audio"].close()
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"语音已保存到: {output_file}")
        return True
    else:
        print(f"合成失败: {response.status_code}")
        return False

# 使用示例
synthesize_with_reference(
    text="这是使用参考音频合成的语音",
    reference_audio_path="reference.wav",
    prompt_text="参考音频对应的文本"
)
```

### 异步调用
```python
import asyncio
import aiohttp

async def async_synthesize(text, language="zh"):
    """异步语音合成"""
    url = "http://localhost:8000/tts"
    payload = {
        "text": text,
        "text_language": language,
        "temperature": 0.6,
        "speed": 1.0
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                content = await response.read()
                with open(f"async_output_{hash(text)}.wav", "wb") as f:
                    f.write(content)
                return True
            else:
                print(f"合成失败: {response.status}")
                return False

# 批量异步合成
async def batch_synthesize(texts):
    """批量异步合成"""
    tasks = [async_synthesize(text) for text in texts]
    results = await asyncio.gather(*tasks)
    return results

# 使用示例
texts = [
    "第一段文本",
    "第二段文本",
    "第三段文本"
]
asyncio.run(batch_synthesize(texts))
```

## JavaScript 示例

### 基础调用
```javascript
async function synthesizeSpeech(text, language = 'zh') {
    try {
        const response = await fetch('http://localhost:8000/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                text_language: language,
                temperature: 0.6,
                speed: 1.0,
                top_k: 20,
                top_p: 0.6
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            
            // 播放音频
            const audio = new Audio(url);
            audio.play();
            
            return url;
        } else {
            console.error('合成失败:', response.status);
            return null;
        }
    } catch (error) {
        console.error('请求失败:', error);
        return null;
    }
}

// 使用示例
synthesizeSpeech('你好，欢迎使用SonixHub！');
```

### 上传参考音频
```javascript
async function synthesizeWithUpload(text, audioFile, promptText) {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('text_language', 'zh');
    formData.append('refer_audio', audioFile);
    formData.append('prompt_text', promptText);
    formData.append('prompt_language', 'zh');
    formData.append('temperature', '0.6');
    formData.append('speed', '1.0');
    
    try {
        const response = await fetch('http://localhost:8000/tts/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            
            // 创建下载链接
            const link = document.createElement('a');
            link.href = url;
            link.download = 'synthesized_speech.wav';
            link.click();
            
            return url;
        } else {
            console.error('合成失败:', response.status);
            return null;
        }
    } catch (error) {
        console.error('请求失败:', error);
        return null;
    }
}

// 文件上传处理
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        synthesizeWithUpload('测试文本', file, '参考音频文本');
    }
});
```

### Node.js 示例
```javascript
const fetch = require('node-fetch');
const fs = require('fs');
const FormData = require('form-data');

async function synthesizeSpeech(text, outputPath = 'output.wav') {
    try {
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
            const buffer = await response.buffer();
            fs.writeFileSync(outputPath, buffer);
            console.log(`语音已保存到: ${outputPath}`);
            return true;
        } else {
            console.error('合成失败:', response.status);
            return false;
        }
    } catch (error) {
        console.error('请求失败:', error);
        return false;
    }
}

// 使用示例
synthesizeSpeech('你好，这是Node.js调用示例');
```

## Java 示例

### 基础调用
```java
import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class SonixHubClient {
    private static final String BASE_URL = "http://localhost:8000";
    private final HttpClient client;
    
    public SonixHubClient() {
        this.client = HttpClient.newHttpClient();
    }
    
    public boolean synthesizeSpeech(String text, String language, String outputPath) {
        try {
            String jsonPayload = String.format(
                "{\"text\":\"%s\",\"text_language\":\"%s\",\"temperature\":0.6,\"speed\":1.0}",
                text, language
            );
            
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(BASE_URL + "/tts"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                    .build();
            
            HttpResponse<byte[]> response = client.send(request, 
                    HttpResponse.BodyHandlers.ofByteArray());
            
            if (response.statusCode() == 200) {
                Path path = Paths.get(outputPath);
                Files.write(path, response.body());
                System.out.println("语音已保存到: " + outputPath);
                return true;
            } else {
                System.err.println("合成失败: " + response.statusCode());
                return false;
            }
        } catch (Exception e) {
            System.err.println("请求失败: " + e.getMessage());
            return false;
        }
    }
    
    public static void main(String[] args) {
        SonixHubClient client = new SonixHubClient();
        client.synthesizeSpeech("你好，这是Java调用示例", "zh", "output.wav");
    }
}
```

### Spring Boot 集成
```java
@RestController
@RequestMapping("/api/tts")
public class TTSController {
    
    private final RestTemplate restTemplate;
    
    public TTSController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }
    
    @PostMapping("/synthesize")
    public ResponseEntity<byte[]> synthesize(@RequestBody TTSRequest request) {
        try {
            String url = "http://localhost:8000/tts";
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<TTSRequest> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    url, HttpMethod.POST, entity, byte[].class);
            
            if (response.getStatusCode() == HttpStatus.OK) {
                HttpHeaders responseHeaders = new HttpHeaders();
                responseHeaders.setContentType(MediaType.parseMediaType("audio/wav"));
                responseHeaders.setContentDisposition(
                        ContentDisposition.builder("attachment")
                                .filename("speech.wav")
                                .build()
                );
                
                return ResponseEntity.ok()
                        .headers(responseHeaders)
                        .body(response.getBody());
            } else {
                return ResponseEntity.status(response.getStatusCode()).build();
            }
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}

// 请求DTO
public class TTSRequest {
    private String text;
    private String text_language = "zh";
    private double temperature = 0.6;
    private double speed = 1.0;
    private int top_k = 20;
    private double top_p = 0.6;
    
    // getters and setters...
}
```

## PHP 示例

### 基础调用
```php
<?php
function synthesizeSpeech($text, $language = 'zh', $outputFile = 'output.wav') {
    $url = 'http://localhost:8000/tts';
    $data = [
        'text' => $text,
        'text_language' => $language,
        'temperature' => 0.6,
        'speed' => 1.0,
        'top_k' => 20,
        'top_p' => 0.6
    ];
    
    $options = [
        'http' => [
            'header' => "Content-Type: application/json\r\n",
            'method' => 'POST',
            'content' => json_encode($data)
        ]
    ];
    
    $context = stream_context_create($options);
    $result = file_get_contents($url, false, $context);
    
    if ($result !== FALSE) {
        file_put_contents($outputFile, $result);
        echo "语音已保存到: $outputFile\n";
        return true;
    } else {
        echo "合成失败\n";
        return false;
    }
}

// 使用示例
synthesizeSpeech('你好，这是PHP调用示例');
?>
```

### 使用cURL
```php
<?php
function synthesizeWithCurl($text, $language = 'zh') {
    $url = 'http://localhost:8000/tts';
    $data = [
        'text' => $text,
        'text_language' => $language,
        'temperature' => 0.6,
        'speed' => 1.0
    ];
    
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json',
        'Content-Length: ' . strlen(json_encode($data))
    ]);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    if ($httpCode === 200) {
        file_put_contents('output.wav', $response);
        echo "语音合成成功\n";
        return true;
    } else {
        echo "合成失败: HTTP $httpCode\n";
        return false;
    }
}

// 使用示例
synthesizeWithCurl('你好，这是cURL调用示例');
?>
```

## Go 示例

### 基础调用
```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
)

type TTSRequest struct {
    Text         string  `json:"text"`
    TextLanguage string  `json:"text_language"`
    Temperature  float64 `json:"temperature"`
    Speed        float64 `json:"speed"`
    TopK         int     `json:"top_k"`
    TopP         float64 `json:"top_p"`
}

func synthesizeSpeech(text, language, outputPath string) error {
    url := "http://localhost:8000/tts"
    
    request := TTSRequest{
        Text:         text,
        TextLanguage: language,
        Temperature:  0.6,
        Speed:        1.0,
        TopK:         20,
        TopP:         0.6,
    }
    
    jsonData, err := json.Marshal(request)
    if err != nil {
        return err
    }
    
    resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("合成失败: %d", resp.StatusCode)
    }
    
    // 保存音频文件
    file, err := os.Create(outputPath)
    if err != nil {
        return err
    }
    defer file.Close()
    
    _, err = io.Copy(file, resp.Body)
    if err != nil {
        return err
    }
    
    fmt.Printf("语音已保存到: %s\n", outputPath)
    return nil
}

func main() {
    err := synthesizeSpeech("你好，这是Go调用示例", "zh", "output.wav")
    if err != nil {
        fmt.Printf("错误: %v\n", err)
    }
}
```

## C# 示例

### 基础调用
```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

public class SonixHubClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    
    public SonixHubClient(string baseUrl = "http://localhost:8000")
    {
        _httpClient = new HttpClient();
        _baseUrl = baseUrl;
    }
    
    public async Task<bool> SynthesizeSpeechAsync(string text, string language = "zh", string outputPath = "output.wav")
    {
        try
        {
            var request = new
            {
                text = text,
                text_language = language,
                temperature = 0.6,
                speed = 1.0,
                top_k = 20,
                top_p = 0.6
            };
            
            var json = JsonSerializer.Serialize(request);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            
            var response = await _httpClient.PostAsync($"{_baseUrl}/tts", content);
            
            if (response.IsSuccessStatusCode)
            {
                var audioBytes = await response.Content.ReadAsByteArrayAsync();
                await File.WriteAllBytesAsync(outputPath, audioBytes);
                Console.WriteLine($"语音已保存到: {outputPath}");
                return true;
            }
            else
            {
                Console.WriteLine($"合成失败: {response.StatusCode}");
                return false;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"请求失败: {ex.Message}");
            return false;
        }
    }
    
    public void Dispose()
    {
        _httpClient?.Dispose();
    }
}

// 使用示例
class Program
{
    static async Task Main(string[] args)
    {
        var client = new SonixHubClient();
        await client.SynthesizeSpeechAsync("你好，这是C#调用示例");
        client.Dispose();
    }
}
```

## 错误处理和重试机制

### Python重试示例
```python
import requests
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"尝试 {attempt + 1} 失败: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def synthesize_with_retry(text, language="zh"):
    """带重试机制的语音合成"""
    url = "http://localhost:8000/tts"
    payload = {
        "text": text,
        "text_language": language,
        "temperature": 0.6,
        "speed": 1.0
    }
    
    response = requests.post(url, json=payload, timeout=30)
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

# 使用示例
try:
    audio_data = synthesize_with_retry("测试重试机制")
    if audio_data:
        with open("retry_output.wav", "wb") as f:
            f.write(audio_data)
        print("合成成功")
except Exception as e:
    print(f"最终失败: {e}")
```

## 性能优化建议

1. **连接池使用**：使用HTTP连接池减少连接开销
2. **异步处理**：对于批量合成，使用异步请求
3. **缓存机制**：对相同文本进行缓存
4. **参数调优**：根据需求调整temperature、speed等参数
5. **错误处理**：实现重试机制和错误恢复
