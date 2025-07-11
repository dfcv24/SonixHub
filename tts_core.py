"""
TTS核心模块
从GPT-SoVITS中提取的核心语音合成功能
"""

import os
import sys
import json
import re
import warnings
import traceback
from typing import Optional, Union, List, Tuple
from time import time as ttime

import torch
import numpy as np
import librosa
import torchaudio
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 屏蔽警告
warnings.filterwarnings("ignore")

# 导入路径配置
from path_config import get_base_paths, get_model_paths, setup_environment

# 设置环境变量
setup_environment()

# 添加GPT-SoVITS路径
try:
    base_paths = get_base_paths()
    gpt_sovits_dir = base_paths["gpt_sovits_dir"]
    
    sys.path.insert(0, gpt_sovits_dir)
    sys.path.insert(0, os.path.join(gpt_sovits_dir, "GPT_SoVITS"))
    sys.path.insert(0, os.path.join(gpt_sovits_dir, "GPT_SoVITS", "eres2net"))
    

except Exception as e:
    print(f"路径配置错误: {e}")
    sys.exit(1)

os.environ["cnhubert_base_path"] = base_paths["cnhubert_path"]
print("CNHubert基础路径已设置:", os.environ["cnhubert_base_path"])

# 导入GPT-SoVITS模块
from feature_extractor import cnhubert
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.models import SynthesizerTrn, SynthesizerTrnV3, Generator
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

from gpt_sovits_fixed.inference_webui_fixed import get_phones_and_bert, get_spepc
from gpt_sovits_fixed.cleaner_fixed import clean_text
from gpt_sovits_fixed.sv_fixed import SV


# 尝试导入BigVGAN
try:
    from BigVGAN import bigvgan
    BIGVGAN_AVAILABLE = True
except ImportError:
    BIGVGAN_AVAILABLE = False



class DictToAttrRecursive(dict):
    """递归字典到属性转换器"""
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)


class TTSCore:
    """TTS核心类"""
    
    def __init__(self, 
                 gpt_path: str = None, 
                 sovits_path: str = None,
                 device: str = "auto",
                 is_half: bool = True):
        """
        初始化TTS核心
        
        Args:
            gpt_path: GPT模型路径
            sovits_path: SoVITS模型路径
            device: 设备 ('cuda', 'cpu', 'auto')
            is_half: 是否使用半精度
        """
        self.device = self._get_device(device)
        self.is_half = is_half and torch.cuda.is_available()
        
        # 语言映射
        self.dict_language = {
            "中文": "all_zh",
            "英文": "en", 
            "日文": "all_ja",
            "中英混合": "zh",
            "日英混合": "ja",
            "多语种混合": "auto",
        }
        
        # 标点符号
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
        
        # 模型变量
        self.ssl_model = None
        self.bert_model = None
        self.tokenizer = None
        self.t2s_model = None
        self.vq_model = None
        self.hps = None
        self.version = None
        self.model_version = None
        self.bigvgan_model = None
        self.hifigan_model = None
        self.sv_cn_model = None
        
        # 加载模型
        self._load_models(gpt_path, sovits_path)
        
        # 缓存
        self.cache = {}
        
    def _get_device(self, device: str) -> str:
        """获取设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_models(self, gpt_path: str, sovits_path: str):
        """加载模型"""
        # 使用配置的模型路径
        try:
            model_paths = get_model_paths()
            
            # 如果没有指定路径，使用配置中的路径
            if not gpt_path:
                gpt_path = model_paths["gpt_path"]
            if not sovits_path:
                sovits_path = model_paths["sovits_path"]
                
            cnhubert_base_path = model_paths["cnhubert_path"]
            bert_path = model_paths["bert_path"]
            
            print(f"使用GPT模型: {os.path.basename(gpt_path)}")
            print(f"使用SoVITS模型: {os.path.basename(sovits_path)}")
            print(f"CNHubert: {'已配置' if os.path.exists(cnhubert_base_path) else '未找到'}")
            print(f"BERT: {'已配置' if os.path.exists(bert_path) else '未找到'}")
            
        except Exception as e:
            print(f"获取模型路径失败: {e}")
            return
            
        # 加载BERT模型
        self._load_bert_model(bert_path)
        
        # 加载SSL模型
        self._load_ssl_model(cnhubert_base_path)
        
        # 加载GPT模型
        if os.path.exists(gpt_path):
            self._load_gpt_model(gpt_path)
        else:
            print(f"GPT模型文件不存在: {gpt_path}")
            
        # 加载SoVITS模型
        if os.path.exists(sovits_path):
            self._load_sovits_model(sovits_path)
        else:
            print(f"SoVITS模型文件不存在: {sovits_path}")
    
    def _load_bert_model(self, bert_path: str):
        """加载BERT模型"""
        if not os.path.exists(bert_path):
            print(f"BERT模型路径不存在: {bert_path}")
            return
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
            if self.is_half:
                self.bert_model = self.bert_model.half()
            self.bert_model = self.bert_model.to(self.device)
            print("BERT模型加载成功")
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
    
    def _load_ssl_model(self, cnhubert_base_path: str):
        """加载SSL模型"""
        try:
            # 设置cnhubert的基础路径
            cnhubert.cnhubert_base_path = cnhubert_base_path
            self.ssl_model = cnhubert.get_model()
            if self.is_half:
                self.ssl_model = self.ssl_model.half()
            self.ssl_model = self.ssl_model.to(self.device)
            print("SSL模型加载成功")
        except Exception as e:
            print(f"加载SSL模型失败: {e}")
            print(f"请检查cnhubert路径: {cnhubert_base_path}")
            # 尝试默认路径
            try:
                self.ssl_model = cnhubert.get_model()
                if self.is_half:
                    self.ssl_model = self.ssl_model.half()
                self.ssl_model = self.ssl_model.to(self.device)
                print("使用默认cnhubert模型加载成功")
            except Exception as e2:
                print(f"使用默认路径也失败: {e2}")
                self.ssl_model = None
    
    def _load_gpt_model(self, gpt_path: str):
        """加载GPT模型"""
        try:
            dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
            config = dict_s1["config"]
            self.hz = 50
            self.max_sec = config["data"]["max_sec"]
            self.t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
            self.t2s_model.load_state_dict(dict_s1["weight"])
            if self.is_half:
                self.t2s_model = self.t2s_model.half()
            self.t2s_model = self.t2s_model.to(self.device)
            self.t2s_model.eval()
            print("GPT模型加载成功")
        except Exception as e:
            print(f"加载GPT模型失败: {e}")
    
    def _load_sovits_model(self, sovits_path: str):
        """加载SoVITS模型"""
        try:
            # 获取模型版本信息 - 使用GPT-SoVITS的原生函数
            try:
                self.version, self.model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
                print(f"检测到模型: {self.model_version} ({self.version})")
            except Exception as e:
                print(f"版本检测失败，使用默认版本: {e}")
                self.version = "v2"
                self.model_version = "v2Pro"
            
            # 加载模型权重
            dict_s2 = load_sovits_new(sovits_path)
            self.hps = dict_s2["config"]
            self.hps = DictToAttrRecursive(self.hps)
            self.hps.model.semantic_frame_rate = "25hz"
            
            # 确定版本（保持原有的版本检测逻辑作为备份）
            if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
                detected_version = "v2"
            elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
                detected_version = "v1"
            else:
                detected_version = "v2"
            
            # 如果自动检测失败，使用权重分析的结果
            if self.version is None:
                self.version = detected_version
                
            self.hps.model.version = self.model_version
            print(f"最终版本: {self.model_version}")
            
            # 创建模型
            if self.model_version in {"v3", "v4"}:
                self.hps.model.version = self.model_version
                self.vq_model = SynthesizerTrnV3(
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    n_speakers=self.hps.data.n_speakers,
                    **self.hps.model,
                )
            else:
                self.vq_model = SynthesizerTrn(
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    n_speakers=self.hps.data.n_speakers,
                    **self.hps.model,
                )
            
            # 加载权重
            if self.is_half:
                self.vq_model = self.vq_model.half()
            self.vq_model = self.vq_model.to(self.device)
            self.vq_model.eval()
            
            # 加载状态字典
            self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
            
            # 初始化声码器
            self._init_vocoder()
            
            print(f"SoVITS模型加载成功 (版本: {self.model_version})")
            
        except Exception as e:
            print(f"加载SoVITS模型失败: {e}")
            traceback.print_exc()
    
    def _init_vocoder(self):
        """初始化声码器"""
        if self.model_version == "v3" and BIGVGAN_AVAILABLE:
            self._init_bigvgan()
        elif self.model_version == "v4":
            self._init_hifigan()
        elif self.model_version in {"v2Pro", "v2ProPlus"}:
            self._init_sv_cn()
    
    def _init_bigvgan(self):
        """初始化BigVGAN"""
        try:
            pretrained_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "GPT-SoVITS", "GPT_SoVITS", "pretrained_models", 
                "models--nvidia--bigvgan_v2_24khz_100band_256x"
            )
            if os.path.exists(pretrained_dir):
                self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                    pretrained_dir, use_cuda_kernel=False
                )
                self.bigvgan_model.remove_weight_norm()
                self.bigvgan_model = self.bigvgan_model.eval()
                if self.is_half:
                    self.bigvgan_model = self.bigvgan_model.half()
                self.bigvgan_model = self.bigvgan_model.to(self.device)
                print("BigVGAN声码器加载成功")
        except Exception as e:
            print(f"加载BigVGAN失败: {e}")
    
    def _init_hifigan(self):
        """初始化HiFiGAN"""
        try:
            self.hifigan_model = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            
            vocoder_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "GPT-SoVITS", "GPT_SoVITS", "pretrained_models", 
                "gsv-v4-pretrained", "vocoder.pth"
            )
            
            if os.path.exists(vocoder_path):
                state_dict_g = torch.load(vocoder_path, map_location="cpu", weights_only=False)
                self.hifigan_model.load_state_dict(state_dict_g)
                self.hifigan_model.eval()
                self.hifigan_model.remove_weight_norm()
                if self.is_half:
                    self.hifigan_model = self.hifigan_model.half()
                self.hifigan_model = self.hifigan_model.to(self.device)
                print("HiFiGAN声码器加载成功")
        except Exception as e:
            print(f"加载HiFiGAN失败: {e}")
    
    def _init_sv_cn(self):
        """初始化SV-CN"""
        try:
            self.sv_cn_model = SV(self.device, self.is_half)
            print("SV-CN模型加载成功")
        except Exception as e:
            print(f"加载SV-CN失败: {e}")
    
    def get_bert_feature(self, text: str, word2ph: List[int]) -> torch.Tensor:
        """获取BERT特征"""
        if not self.bert_model or not self.tokenizer:
            # 根据版本返回正确维度的零tensor
            bert_dim = 1024 if self.version == "v1" else 768
            return torch.zeros(
                (bert_dim, sum(word2ph)), 
                dtype=torch.float16 if self.is_half else torch.float32
            ).to(self.device)
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                res = self.bert_model(**inputs, output_hidden_states=True)
                
                # 根据版本选择层
                if self.version == "v1":
                    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
                else:
                    res = res["hidden_states"][-1][0].cpu()[1:-1]
            
            assert len(word2ph) == len(text), f"word2ph length {len(word2ph)} != text length {len(text)}"
            
            phone_level_feature = []
            for i in range(len(word2ph)):
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)
            
            phone_level_feature = torch.cat(phone_level_feature, dim=0)
            return phone_level_feature.T.to(self.device)
                
        except Exception as e:
            print(f"BERT特征提取失败: {e}")
            bert_dim = 1024 if self.version == "v1" else 768
            return torch.zeros(
                (bert_dim, sum(word2ph)), 
                dtype=torch.float16 if self.is_half else torch.float32
            ).to(self.device)
    
    def synthesize(self, 
                   text: str,
                   text_language: str = "zh",
                   refer_wav_path: str = None,
                   prompt_text: str = None,
                   prompt_language: str = "zh",
                   top_k: int = 20,
                   top_p: float = 0.6,
                   temperature: float = 0.6,
                   speed: float = 1.0,
                   ref_free: bool = False,
                   **kwargs) -> Tuple[int, np.ndarray]:
        """
        语音合成 - 完全参考inference_webui.py实现
        """
        if not self.vq_model or not self.t2s_model:
            raise RuntimeError("模型未正确加载")
        
        if not refer_wav_path or not os.path.exists(refer_wav_path):
            raise ValueError("参考音频路径无效")
        
        # 处理语言映射
        prompt_language = self.dict_language.get(prompt_language, prompt_language)
        text_language = self.dict_language.get(text_language, text_language)
        
        # 检查是否使用无参考文本模式
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        
        # 处理参考音频
        if not ref_free:
            with torch.no_grad():
                wav16k, sr = librosa.load(refer_wav_path, sr=16000)
                wav16k = torch.from_numpy(wav16k)
                if self.is_half:
                    wav16k = wav16k.half()
                wav16k = wav16k.to(self.device)
                
                ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(self.device)
        
        # 处理文本 - 按照原始逻辑
        if not ref_free:
            # 处理参考文本
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in self.splits:
                prompt_text += "。" if prompt_language != "en" else "."
            phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, self.version)
        
        # 处理目标文本
        text = text.strip("\n")
        if text[-1] not in self.splits:
            text += "。" if text_language != "en" else "."
        
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, self.version)
        
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
        
        bert = bert.to(self.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
        
        # GPT推理
        with torch.no_grad():
            pred_semantic, idx = self.t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=self.hz * self.max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        
        # 音频解码
        dtype = torch.float32 if not self.is_half else torch.float16
        refers, refer_audio = get_spepc(self.hps, refer_wav_path, dtype, self.device, True)
        refers = [refers]
        # 计算说话人嵌入
        with torch.no_grad():
            sv_emb = [self.sv_cn_model.compute_embedding3(refer_audio)]
        
        # 为v2pro和v2ProPlus模型计算sv_emb
        if self.model_version in {"v2Pro", "v2ProPlus"} and self.sv_cn_model:            
            audio = self.vq_model.decode(
                pred_semantic, 
                torch.LongTensor(phones2).to(self.device).unsqueeze(0), 
                refers, 
                speed=speed,
                sv_emb=sv_emb
            )[0][0]
        else:
            # 对于其他版本，不使用sv_emb
            audio = self.vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refers, speed=speed
            )[0][0]
        
        # 音量归一化
        max_audio = torch.abs(audio).max()
        if max_audio > 1:
            audio = audio / max_audio
        
        # 确定采样率
        if self.model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
            sr = 32000
        elif self.model_version == "v3":
            sr = 24000
        else:
            sr = 48000
        
        # 转换为numpy数组
        audio_np = audio.cpu().detach().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        return sr, audio_int16
    
    def get_bert_inf(self, phones: List[int], word2ph: List[int], norm_text: str, language: str) -> torch.Tensor:
        """获取BERT推理特征 - 参考原始inference_webui.py实现"""
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph)
        else:
            bert_dim = 1024 if self.version == "v1" else 768
            bert = torch.zeros(
                (bert_dim, len(phones)),
                dtype=torch.float16 if self.is_half else torch.float32,
            ).to(self.device)
        return bert
    
    # ...existing code...


# 全局TTS实例管理
_tts_instance = None


def get_tts_instance(**kwargs) -> TTSCore:
    """获取TTS实例（单例模式）"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSCore(**kwargs)
    return _tts_instance


def synthesize_speech(text: str, **kwargs) -> Tuple[int, np.ndarray]:
    """便捷的语音合成函数"""
    tts = get_tts_instance()
    return tts.synthesize(text, **kwargs)


if __name__ == "__main__":
    # 测试代码
    tts = TTSCore()
    print("TTS核心模块初始化完成")
    print(f"设备: {tts.device}")
    print(f"模型版本: {tts.model_version}")
    print(f"支持的语言: {list(tts.dict_language.keys())}")
