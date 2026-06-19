"""
RH LLM API Node — OpenAI 兼容接口
=====================================
基于 ComfyUI_RH_LLM_API 架构，集成 Gemma4 的预设模板体系。
支持纯文本对话、图像理解和视频理解（通过 base64 编码）。

使用方式:
    RhLlmApiNode (API 调用) ──(STRING)──> 下游节点
                                    ├─ 有图: 图像理解
                                    ├─ 有视频: 视频理解
                                    └─ 无图/视频: 纯文本对话
"""

import os
import sys
import time
import re
import base64
import io

import torch
import numpy as np
from PIL import Image


# ============================================================
# 1. 工具函数
# ============================================================

def _has_valid_image(image):
    if image is None:
        return False
    if not isinstance(image, torch.Tensor):
        return False
    if image.numel() == 0:
        return False
    return True


def encode_image_b64(ref_image):
    """
    将 ComfyUI IMAGE tensor 编码为 base64 JPEG（不缩放）。
    兼容 OpenAI vision API 格式。
    """
    i = 255.0 * ref_image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_video_file_path(video):
    """从 ComfyUI VIDEO 对象中提取文件路径。"""
    if hasattr(video, "_VideoFromFile__file"):
        path = getattr(video, "_VideoFromFile__file", None)
        if isinstance(path, str) and os.path.exists(path):
            return path
    if hasattr(video, "get_stream_source"):
        try:
            stream_source = video.get_stream_source()
            if isinstance(stream_source, str) and os.path.exists(stream_source):
                return stream_source
        except Exception:
            pass
    for attr in ("path", "file"):
        if hasattr(video, attr):
            path = getattr(video, attr, None)
            if isinstance(path, str) and os.path.exists(path):
                return path
    return None


def encode_video_b64(video):
    """
    将 ComfyUI VIDEO 对象编码为 base64 MP4。
    优先读取文件路径，避免临时文件。
    """
    video_path = _get_video_file_path(video)
    if video_path:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    if hasattr(video, "save_to"):
        temp_path = f"temp_video_{time.time()}.mp4"
        try:
            video.save_to(temp_path)
            with open(temp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    raise ValueError(f"无法读取 VIDEO 数据，对象类型: {type(video)}")


def _clean_prompt_output(text):
    """清洗输出中的指令性前缀，保持纯提示词。"""
    if not text:
        return text
    text = re.sub(
        r"^(这是|以下|根据|Here is|Here's|Below is|好的|明白)[^。\n]*[。:\n]",
        "", text, count=3, flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(r"\n*[-=]{3,}.*$", "", text, flags=re.DOTALL)
    return text.strip()


# ============================================================
# 2. 预设模板（复用 Gemma4 presets）
# ============================================================

from .presets import (
    _EXPAND_PRESETS, _VISION_PRESETS,
    _ALL_PRESET_NAMES, _MERGED_MAP,
)


# ============================================================
# 3. 节点定义
# ============================================================

class RhLlmApiNode:
    """
    API LLM 节点 — 同时支持纯文本、图像理解和视频理解。
    集成 Gemma4 预设模板，支持 OpenAI 兼容 API（DeepSeek、Qwen 等）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {
                    "default": "https://api.openai.com/v1",
                    "placeholder": "https://api.openai.com/v1 或其他 OpenAI 兼容端点",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "sk-...（留空则使用环境变量 OPENAI_API_KEY）",
                }),
                "model": ("STRING", {
                    "default": "gpt-4o",
                    "placeholder": "gpt-4o / deepseek-chat / qwen-vl-max 等",
                }),
                "system_preset": (_ALL_PRESET_NAMES, {
                    "default": _ALL_PRESET_NAMES[0],
                    "tooltip": "留空 system_prompt 时使用预设模板",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 32,
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义 System Prompt（留空则使用上方预设）",
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "User Prompt / 问题描述（选填）",
                }),
                "image": ("IMAGE",),
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = False
    CATEGORY = "AI/LLM API"
    FUNCTION = "call_api"

    def call_api(self, api_baseurl, api_key, model, system_preset,
                 temperature, max_tokens, system_prompt="",
                 user_prompt="", image=None, video=None):
        # 构建最终 system prompt
        final_system = (
            system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else _MERGED_MAP.get(system_preset, "")
        )

        # 检测输入类型
        has_image = _has_valid_image(image)
        has_video = video is not None

        print(f"[RhLlmApi] 调用 API: baseurl={api_baseurl}  model={model}  "
              f"image={'YES' if has_image else 'NO'}  video={'YES' if has_video else 'NO'}")

        # 构建消息
        if has_video:
            base64_video = encode_video_b64(video)
            messages = [
                {"role": "system", "content": final_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt or "描述这段视频内容。"},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{base64_video}"
                            },
                        },
                    ],
                },
            ]
        elif has_image:
            base64_image = encode_image_b64(image)
            messages = [
                {"role": "system", "content": final_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt or "描述这张图片。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ]
        else:
            messages = [
                {"role": "system", "content": final_system},
                {"role": "user", "content": user_prompt or "Hello"},
            ]

        # 调用 API
        try:
            from openai import OpenAI
        except ImportError as e:
            return ("[错误] 未安装 openai 库。请执行: pip install openai",)

        try:
            # 优先使用传入的 api_key，否则尝试环境变量
            effective_key = api_key.strip() if api_key else None
            client_kwargs = {"base_url": api_baseurl}
            if effective_key:
                client_kwargs["api_key"] = effective_key
            # 否则 openai 会自动读取 OPENAI_API_KEY 环境变量

            client = OpenAI(**client_kwargs)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_output = completion.choices[0].message.content
            result = _clean_prompt_output(raw_output) if raw_output else ""
            print(f"[RhLlmApi] ✓ 调用成功，输出长度: {len(result)}")
            return (result,)
        except Exception as e:
            error_msg = str(e)
            print(f"[RhLlmApi] ✗ 调用失败: {error_msg}")
            return (f"[API 调用失败] {error_msg}",)


# ============================================================
# 4. 节点注册（由 __init__.py 导入使用）
# ============================================================

NODE_CLASS_MAPPINGS = {
    "RhLlmApiNode": RhLlmApiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RhLlmApiNode": "RH LLM API Node    [API调用:图/视频/文本]",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
