"""
RH LLM API Node — OpenAI 兼容接口
=====================================
基于 ComfyUI_RH_LLM_API 架构，集成 Gemma4 的预设模板体系。
支持纯文本对话、图像理解（多图并发）和视频理解（通过提取帧发送）。

使用方式:
    RhLlmApiNode (API 调用) ──(STRING)──> 下游节点
                                    ├─ 有图: 图像理解（多图并发）
                                    ├─ 有视频: 视频帧提取后发送
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


def encode_images_b64(ref_image):
    """
    将 ComfyUI IMAGE tensor 编码为 base64 JPEG 列表（支持多图批量）。
    IMAGE 格式: [B, H, W, 3]，B=batch_size=图的数量。
    返回: list[str] —— 每个元素是一张图的 base64 字符串。
    """
    b64_list = []
    if ref_image.dim() == 4:
        n = ref_image.shape[0]
        for i in range(n):
            arr = (ref_image[i].cpu().numpy() * 255).astype("uint8")
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90, optimize=True)
            b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    else:
        arr = (ref_image.cpu().numpy() * 255).astype("uint8")
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return b64_list


def encode_image_b64(ref_image):
    """向后兼容：只取第一张图，返回单个 base64 字符串。"""
    return encode_images_b64(ref_image)[0]


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
    for attr in ("path", "file", "file_path", "filepath", "info", "_path"):
        if hasattr(video, attr):
            path = getattr(video, attr, None)
            if isinstance(path, str) and os.path.exists(path):
                return path
    return None


def _extract_video_frames(video, n_frames=8):
    """
    从 ComfyUI VIDEO 对象中提取关键帧，返回 PIL.Image 列表。
    优先用 imageio；失败时尝试 decord。
    """
    video_path = _get_video_file_path(video)
    if not video_path:
        print(f"[RhLlmApi] 无法获取视频文件路径")
        return []

    # 1. 用 imageio 提取
    try:
        import imageio
        reader = imageio.get_reader(video_path, 'ffmpeg')
        try:
            total = reader.count_frames()
        except Exception:
            total = 0
            try:
                meta = reader.get_meta_data()
                if meta.get('duration', 0) > 0 and meta.get('fps', 0) > 0:
                    total = int(meta['duration'] * meta['fps'])
            except Exception:
                pass
        if total <= 0:
            total = 100

        actual_n = min(n_frames, total)
        indices = [int(i * total / actual_n) for i in range(actual_n)]

        frames = []
        for idx in indices:
            try:
                frame = reader.get_data(idx)
                frames.append(Image.fromarray(frame))
            except Exception:
                continue
        reader.close()
        if frames:
            return frames
    except ImportError:
        pass
    except Exception as e:
        print(f"[RhLlmApi] imageio 视频帧提取失败: {e}")

    # 2. 回退：用 decord
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = [int(i * total / n_frames) for i in range(n_frames)]
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception as e:
        print(f"[RhLlmApi] decord 视频帧提取失败: {e}")
        return []
    return []


def _pil_images_to_b64(pil_images):
    """将 PIL.Image 列表转 base64 JPEG 列表。"""
    b64_list = []
    for img in pil_images:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return b64_list


def encode_video_b64(video):
    """
    将 ComfyUI VIDEO 对象编码为 base64 MP4（整个视频发送）。
    优先读取文件路径，避免临时文件。
    
    注：大多数 API 后端不支持直接发送整个视频文件，而是支持的发送多帧 image_url 更通用。
    这个函数保留用于向后兼容/特殊后端。
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
    API LLM 节点 — 支持纯文本、多图理解和视频理解（提取帧后发送）。
    集成 Gemma4 预设模板，支持 OpenAI 兼容 API（DeepSeek、Qwen 等）。
    
    视频理解原理：
        - 视频 → 提取 N 帧关键帧 → 每张图按时间顺序作为多个 image_url 消息项发送
        - 比直接发送整个 MP4 文件更兼容大多数 API 后端
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
                "video_frames": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = False
    CATEGORY = "AI/LLM API"
    FUNCTION = "call_api"

    def call_api(self, api_baseurl, api_key, model, system_preset,
                 temperature, max_tokens, system_prompt="",
                 user_prompt="", image=None, video=None, video_frames=8):
        # 构建最终 system prompt
        final_system = (
            system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else _MERGED_MAP.get(system_preset, "")
        )

        # 检测输入类型
        has_image = _has_valid_image(image)
        has_video = video is not None

        # 收集所有 base64 图像（视频帧 + 普通图像）
        all_b64_images = []

        if has_video:
            frames = _extract_video_frames(video, n_frames=video_frames)
            if frames:
                all_b64_images.extend(_pil_images_to_b64(frames))
                print(f"[RhLlmApi] 视频输入：提取 {len(frames)} 帧")

        if has_image:
            img_b64_list = encode_images_b64(image)
            all_b64_images.extend(img_b64_list)
            print(f"[RhLlmApi] 图像输入：{len(img_b64_list)} 张")

        has_any_image = len(all_b64_images) > 0

        print(f"[RhLlmApi] 调用 API: baseurl={api_baseurl}  model={model}  "
              f"images={len(all_b64_images)}  video={'YES' if has_video else 'NO'}")

        # 构建消息
        if has_any_image:
            # 多图/视频帧：按顺序放入多个 image_url 项
            image_urls = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                for b64 in all_b64_images
            ]
            messages = [
                {"role": "system", "content": final_system},
                {
                    "role": "user",
                    "content": image_urls + [
                        {"type": "text", "text": user_prompt or ("描述这些图像/视频内容。" if len(all_b64_images) > 1 else "描述这张图片。")}
                    ],
                },
            ]
        else:
            # 纯文本
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
            # 优先使用传入的 api_key，否则使用环境变量
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
