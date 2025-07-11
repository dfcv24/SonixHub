"""
路径配置文件
统一管理TTS服务的各种路径
"""

import os


def get_base_paths():
    """获取基础路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找GPT-SoVITS目录
    gpt_sovits_dir = None
    possible_dirs = [
        os.path.join(os.path.dirname(current_dir), "GPT-SoVITS"),  # 上级目录
        os.path.join(current_dir, "GPT-SoVITS"),  # 当前目录
        "/home/luzhiwei/git-home/GPT-SoVITS"  # 绝对路径
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            gpt_sovits_dir = dir_path
            break
    
    if not gpt_sovits_dir:
        raise FileNotFoundError("未找到GPT-SoVITS目录")
    
    return {
        "current_dir": current_dir,
        "gpt_sovits_dir": gpt_sovits_dir,
        "pretrained_dir": os.path.join(gpt_sovits_dir, "GPT_SoVITS", "pretrained_models"),
        "cnhubert_path": os.path.join(gpt_sovits_dir, "GPT_SoVITS/pretrained_models/chinese-hubert-base")
    }


def get_model_paths():
    """获取模型路径"""
    paths = get_base_paths()
    gpt_sovits_dir = paths["gpt_sovits_dir"]
    pretrained_dir = paths["pretrained_dir"]
    
    # GPT模型路径
    gpt_paths = [
        # os.path.join(gpt_sovits_dir, "GPT_weights_v2Pro", "test3-e8.ckpt"),
        # os.path.join(gpt_sovits_dir, "GPT_weights", "test3-e8.ckpt"),
        # os.path.join(pretrained_dir, "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
        os.path.join(pretrained_dir, "s1v3.ckpt")
    ]
    
    # SoVITS模型路径
    sovits_paths = [
        os.path.join(gpt_sovits_dir, "SoVITS_weights_v2Pro", "test3_e8_s376.pth"),
        os.path.join(gpt_sovits_dir, "SoVITS_weights", "test3_e8_s376.pth"),
        os.path.join(pretrained_dir, "v2Pro", "s2Gv2Pro.pth")
        # os.path.join(pretrained_dir,"s2G488k.pth"),
    ]
    
    # G2PW模型路径
    g2pw_paths = [
        os.path.join(gpt_sovits_dir, "GPT_SoVITS", "text", "G2PWModel"),
        os.path.join(pretrained_dir, "G2PWModel")
    ]
    
    # 查找存在的模型文件
    gpt_path = None
    for path in gpt_paths:
        if os.path.exists(path):
            gpt_path = path
            break
    
    sovits_path = None
    for path in sovits_paths:
        if os.path.exists(path):
            sovits_path = path
            break
    
    g2pw_path = None
    for path in g2pw_paths:
        if os.path.exists(path):
            g2pw_path = path
            break
    
    return {
        "gpt_path": gpt_path,
        "sovits_path": sovits_path,
        "cnhubert_path": os.path.join(pretrained_dir, "chinese-hubert-base"),
        "bert_path": os.path.join(pretrained_dir, "chinese-roberta-wwm-ext-large"),
        "sv_path": os.path.join(pretrained_dir, "sv", "pretrained_eres2netv2w24s4ep4.ckpt"),
        "g2pw_path": g2pw_path
    }


def get_reference_audio_paths():
    """获取参考音频路径"""
    paths = get_base_paths()
    gpt_sovits_dir = paths["gpt_sovits_dir"]
    
    possible_ref_dirs = [
        os.path.join(gpt_sovits_dir, "output", "slicer_opt"),
        os.path.join(gpt_sovits_dir, "output"),
        os.path.join(gpt_sovits_dir, "test_audio")
    ]
    
    ref_audio_files = []
    for ref_dir in possible_ref_dirs:
        if os.path.exists(ref_dir):
            for file in os.listdir(ref_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    ref_audio_files.append(os.path.join(ref_dir, file))
    
    return ref_audio_files


def check_paths():
    """检查所有路径是否存在"""
    print("=== 路径检查 ===")
    
    try:
        base_paths = get_base_paths()
        print(f"✓ GPT-SoVITS目录: {base_paths['gpt_sovits_dir']}")
        print(f"✓ 预训练模型目录: {base_paths['pretrained_dir']}")
        
        model_paths = get_model_paths()
        
        if model_paths['gpt_path']:
            print(f"✓ GPT模型: {model_paths['gpt_path']}")
        else:
            print("✗ GPT模型: 未找到")
        
        if model_paths['sovits_path']:
            print(f"✓ SoVITS模型: {model_paths['sovits_path']}")
        else:
            print("✗ SoVITS模型: 未找到")
        
        if os.path.exists(model_paths['cnhubert_path']):
            print(f"✓ CNHubert模型: {model_paths['cnhubert_path']}")
        else:
            print(f"✗ CNHubert模型: {model_paths['cnhubert_path']}")
        
        if os.path.exists(model_paths['bert_path']):
            print(f"✓ BERT模型: {model_paths['bert_path']}")
        else:
            print(f"✗ BERT模型: {model_paths['bert_path']}")
        
        if os.path.exists(model_paths['sv_path']):
            print(f"✓ SV模型: {model_paths['sv_path']}")
        else:
            print(f"✗ SV模型: {model_paths['sv_path']}")
        
        if model_paths['g2pw_path'] and os.path.exists(model_paths['g2pw_path']):
            print(f"✓ G2PW模型: {model_paths['g2pw_path']}")
        else:
            print(f"✗ G2PW模型: {model_paths['g2pw_path'] or '未找到'}")
        
        ref_files = get_reference_audio_paths()
        if ref_files:
            print(f"✓ 找到 {len(ref_files)} 个参考音频文件")
            for i, file in enumerate(ref_files[:3]):  # 只显示前3个
                print(f"  - {file}")
            if len(ref_files) > 3:
                print(f"  ... 还有 {len(ref_files) - 3} 个文件")
        else:
            print("✗ 未找到参考音频文件")
        
        print("=== 检查完成 ===")
        return True
        
    except Exception as e:
        print(f"✗ 路径检查失败: {e}")
        return False


def setup_environment():
    """设置环境变量"""
    try:
        model_paths = get_model_paths()
        
        # 设置BERT路径环境变量
        if model_paths["bert_path"] and os.path.exists(model_paths["bert_path"]):
            os.environ["bert_path"] = model_paths["bert_path"]
            # print(f"设置bert_path环境变量: {model_paths['bert_path']}")
        
        # 设置G2PW路径环境变量
        if model_paths["g2pw_path"] and os.path.exists(model_paths["g2pw_path"]):
            os.environ["g2pw_model_dir"] = model_paths["g2pw_path"]
            # print(f"设置g2pw_model_dir环境变量: {model_paths['g2pw_path']}")
        
        # 设置其他可能需要的环境变量
        base_paths = get_base_paths()
        os.environ["gpt_sovits_dir"] = base_paths["gpt_sovits_dir"]
        
        return True
        
    except Exception as e:
        print(f"设置环境变量失败: {e}")
        return False


if __name__ == "__main__":
    setup_environment()
    check_paths()
