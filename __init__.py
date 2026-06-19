"""
ComfyUI Gemma4 12B — 图像反推 & 提示词扩写
============================================

标准 ComfyUI 管道式架构:
    Gemma4ModelLoader ──(GEMMA4_MODEL)──> Gemma4ImageCaption (双模式)
                                        ├─ 有图: 图像反推
                                        └─ 无图: 提示词扩写优化

模型目录:
    ComfyUI/models/gemma4/
        gemma-4-12b-it/                       # HuggingFace 格式（目录）
        gemma-4-12b-it-Q4_K_M.gguf           # GGUF 主模型
        gemma-4-12b-it-mmproj-f16.gguf       # GGUF 视觉投影（必需）
"""

import os
import sys
import glob
import re

import torch
from PIL import Image

import folder_paths


# ============================================================
# 1. 注册 ComfyUI 标准模型目录
# ============================================================

GEMMA4_MODEL_DIR = "gemma4"
folder_paths.add_model_folder_path(
    GEMMA4_MODEL_DIR,
    os.path.join(folder_paths.models_dir, GEMMA4_MODEL_DIR),
)


# ============================================================
# 2. 模型扫描与路径解析
# ============================================================

def list_gemma4_models():
    model_entries = []
    gemma4_root = os.path.join(folder_paths.models_dir, GEMMA4_MODEL_DIR)
    if not os.path.exists(gemma4_root):
        return model_entries

    hf_dirs = set()
    for subdir in sorted(os.listdir(gemma4_root)):
        subdir_path = os.path.join(gemma4_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        has_config = os.path.exists(os.path.join(subdir_path, "config.json"))
        has_weights = (
            bool(glob.glob(os.path.join(subdir_path, "*.safetensors")))
            or bool(glob.glob(os.path.join(subdir_path, "*.bin")))
            or bool(glob.glob(os.path.join(subdir_path, "model*.pt")))
            or bool(glob.glob(os.path.join(subdir_path, "pytorch_model*")))
        )
        if has_config and has_weights:
            model_entries.append(f"{subdir} (HF)")
            hf_dirs.add(subdir)

    for fname in sorted(os.listdir(gemma4_root)):
        if fname.lower().endswith(".gguf"):
            model_entries.append(f"{fname} (GGUF)")

    for subdir in sorted(os.listdir(gemma4_root)):
        subdir_path = os.path.join(gemma4_root, subdir)
        if not os.path.isdir(subdir_path) or subdir in hf_dirs:
            continue
        for fname in sorted(os.listdir(subdir_path)):
            if fname.lower().endswith(".gguf"):
                model_entries.append(f"{subdir}/{fname} (GGUF)")

    return model_entries


def resolve_model_path(display_name):
    gemma4_root = os.path.join(folder_paths.models_dir, GEMMA4_MODEL_DIR)
    display_name = display_name.strip()

    if display_name.endswith("(GGUF)"):
        model_format = "GGUF"
        raw_name = display_name[: -len("(GGUF)")].strip()
    elif display_name.endswith("(HF)"):
        model_format = "HF"
        raw_name = display_name[: -len("(HF)")].strip()
    else:
        raw_name = display_name
        candidate_hf = os.path.join(gemma4_root, raw_name)
        if os.path.isdir(candidate_hf) and os.path.exists(
            os.path.join(candidate_hf, "config.json")
        ):
            model_format = "HF"
        elif raw_name.lower().endswith(".gguf") or os.path.isfile(
            os.path.join(gemma4_root, raw_name)
        ):
            model_format = "GGUF"
        else:
            model_format = "HF"

    if model_format == "HF":
        full_path = os.path.join(gemma4_root, raw_name)
        if not os.path.isdir(full_path):
            raise RuntimeError(f"找不到模型目录: {full_path}")
    else:
        full_path = os.path.join(gemma4_root, raw_name)
        if not os.path.isfile(full_path):
            raise RuntimeError(f"找不到 GGUF 模型文件: {full_path}")

    return full_path, model_format


# ============================================================
# 3. 模型加载
# ============================================================

def _load_hf_model(model_dir, device_map="auto", torch_dtype="bfloat16", load_in_4bit=False):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[Gemma4Loader]   加载 Processor ...")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    load_kwargs = {
        "torch_dtype": dtype_map.get(torch_dtype, torch.bfloat16),
        "device_map": device_map,
        "local_files_only": True,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print(f"[Gemma4Loader]   启用 bitsandbytes 4-bit 量化")
        except ImportError:
            print(f"[Gemma4Loader]   未安装 bitsandbytes，跳过 4-bit 量化")

    print(f"[Gemma4Loader]   加载模型: device_map={device_map}  dtype={torch_dtype}")
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_dir, **load_kwargs)
    except Exception as e:
        print(f"[Gemma4Loader]   AutoModelForImageTextToText 不可用: {e}")
        print(f"[Gemma4Loader]   回退尝试 AutoModelForCausalLM ...")
        from transformers import AutoModelForCausalLM
        _ = load_kwargs.pop("quantization_config", None)
        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)

    model.eval()
    return model, processor


def _load_gguf_model(gguf_path, n_gpu_layers=-1, n_ctx=8192):
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise RuntimeError(
            "未安装 llama-cpp-python。\n请执行: pip install llama-cpp-python\n"
            f"(原始错误: {e})"
        )

    gguf_dir = os.path.dirname(gguf_path)
    mmproj_candidates = (
        glob.glob(os.path.join(gguf_dir, "*mmproj*.gguf"))
        + glob.glob(os.path.join(gguf_dir, "*vision*.gguf"))
    )
    all_gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))

    print(
        f"[Gemma4Loader]   目录: {gguf_dir}\n"
        f"[Gemma4Loader]   所有 .gguf 文件 ({len(all_gguf_files)}): "
        f"{[os.path.basename(f) for f in all_gguf_files]}\n"
        f"[Gemma4Loader]   mmproj 候选 ({len(mmproj_candidates)}): "
        f"{[os.path.basename(f) for f in mmproj_candidates] if mmproj_candidates else '(无)'}"
    )

    mmproj_path = mmproj_candidates[0] if mmproj_candidates else None
    is_mm = mmproj_path is not None

    print(
        f"[Gemma4Loader] llama.cpp 初始化: "
        f"model={os.path.basename(gguf_path)}  "
        f"n_gpu_layers={n_gpu_layers}  n_ctx={n_ctx}  "
        f"vision_mmproj={'YES (' + os.path.basename(mmproj_path) + ')' if is_mm else 'NO'}"
    )

    llama_kwargs = {
        "model_path": gguf_path,
        "n_ctx": n_ctx,
        "n_batch": 512,
        "n_gpu_layers": n_gpu_layers,
        "verbose": False,
    }

    if is_mm:
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
        llama_kwargs["chat_handler"] = Gemma4ChatHandler(
            clip_model_path=mmproj_path, verbose=False,
        )
        print(f"[Gemma4Loader]   Gemma4ChatHandler 已配置 (mmproj={os.path.basename(mmproj_path)})")

    try:
        model = Llama(**llama_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"llama-cpp-python 加载失败: {e}\n"
            f"  常见原因:\n"
            f"  - 尝试 CPU 模式验证 (gpu_layers=0)\n"
            f"  - n_ctx 超过模型支持上限\n"
            f"  - Gemma4ChatHandler 缺少 mtmd_cpp 库"
        )

    model._gemma4_cfg = {"is_mm": is_mm, "mmproj_path": mmproj_path, "gguf_dir": gguf_dir}

    if not is_mm:
        warning = (
            "\n" + "=" * 68 + "\n"
            "[Gemma4Loader] ⚠ 未找到 mmproj 视觉投影文件\n"
            f"[Gemma4Loader]   目录: {gguf_dir}\n"
            f"[Gemma4Loader]   目录下 .gguf 文件: {[os.path.basename(f) for f in all_gguf_files]}\n"
            "[Gemma4Loader]   → 纯文本 LLM 模式（图像输入将返回错误提示）\n"
            "[Gemma4Loader]   → 请确保 *mmproj*.gguf 与主模型在同一目录\n"
            "[Gemma4Loader]   → 如果已存在，请重启 ComfyUI 重新加载模型\n"
            + "=" * 68 + "\n"
        )
        print(warning)
        print(warning, file=sys.stderr)

    return model, None


# ============================================================
# 4. 推理接口
# ============================================================

def run_inference(model_obj, processor, model_format, pil_image,
                  system_prompt="", user_prompt="",
                  max_new_tokens=512, temperature=0.7):
    if model_format == "HF":
        return _infer_hf(model_obj, processor, pil_image,
                         system_prompt, user_prompt, max_new_tokens, temperature)
    else:
        return _infer_gguf(model_obj, pil_image,
                           system_prompt, user_prompt, max_new_tokens, temperature)


def run_text_inference(model_obj, processor, model_format,
                       system_prompt="", user_prompt="",
                       max_new_tokens=512, temperature=0.7):
    if model_format == "HF":
        return _infer_hf_text(model_obj, processor,
                              system_prompt, user_prompt, max_new_tokens, temperature)
    else:
        return _infer_gguf_text(model_obj,
                                system_prompt, user_prompt, max_new_tokens, temperature)


def _build_instruction(system_prompt, user_prompt):
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    if user_prompt:
        parts.append(user_prompt)
    return "\n".join(parts) if parts else "Describe this image in detail."


def _infer_hf(model, processor, pil_image, system_prompt, user_prompt,
              max_new_tokens, temperature):
    instruction = _build_instruction(system_prompt, user_prompt)
    user_text = f"<image>\n{instruction}"
    inputs = processor(text=user_text, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
        )
    full_text = processor.decode(outputs[0], skip_special_tokens=True)
    return full_text.strip()


def _infer_hf_text(model, processor, system_prompt, user_prompt,
                   max_new_tokens, temperature):
    instruction = _build_instruction(system_prompt, user_prompt)
    inputs = processor(text=instruction, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
        )
    full_text = processor.decode(outputs[0], skip_special_tokens=True)
    return _clean_prompt_output(full_text)


def _infer_gguf(llama, pil_image, system_prompt, user_prompt,
                max_new_tokens, temperature):
    import base64
    from io import BytesIO

    cfg = getattr(llama, "_gemma4_cfg", {})
    is_mm = cfg.get("is_mm", False)

    if not is_mm:
        return (
            "[Gemma4 提示] 当前 GGUF 模型缺少 mmproj 视觉投影，无法理解图片。\n\n"
            "请将配套的 *mmproj*.gguf 与主模型放在同一目录，重启 ComfyUI。\n"
            "（如文件已在目录中，请关闭 ComfyUI 并重新启动以重新加载模型）"
        )

    buffer = BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG", quality=92)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    instruction = _build_instruction(system_prompt, user_prompt)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": instruction},
        ],
    }]

    resp = llama.create_chat_completion(
        messages=messages, max_tokens=max_new_tokens,
        temperature=temperature, top_p=0.9,
    )
    return resp["choices"][0]["message"]["content"].strip()


def _infer_gguf_text(llama, system_prompt, user_prompt,
                     max_new_tokens, temperature):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt or "Hello"})

    resp = llama.create_chat_completion(
        messages=messages, max_tokens=max_new_tokens,
        temperature=temperature, top_p=0.9,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    return raw


# ============================================================
# 5. 输出清洗
# ============================================================

def _clean_prompt_output(text):
    if not text:
        return text
    text = re.sub(
        r"^(这是|以下|根据|Here is|Here's|Below is|好的|明白)[^。\n]*[。:\n]",
        "", text, count=3, flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(r"\n*[-=]{3,}.*$", "", text, flags=re.DOTALL)
    return text.strip()


# ============================================================
# 6. 工具函数
# ============================================================

def _model_choices():
    available = list_gemma4_models()
    if not available:
        available = ["(未检测到模型，请将模型放入 models/gemma4/)"]
    return (available,)


def _image_to_pil(image_tensor):
    if image_tensor.dim() == 4:
        arr = (image_tensor[0].cpu().numpy() * 255).astype("uint8")
    else:
        arr = (image_tensor.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def _has_valid_image(image):
    if image is None:
        return False
    if not isinstance(image, torch.Tensor):
        return False
    if image.numel() == 0:
        return False
    return True


# ============================================================
# 7b. 动态检测 mmproj 状态 & 自动恢复模型（关键修复）
# 解决：运行一次后 model_obj 被设为 None；添加 mmproj 后不重启也能生效
# ============================================================

def _detect_mmproj(gguf_dir):
    """扫描目录，返回 mmproj 文件路径或 None。每次调用都会重新扫描。"""
    candidates = (
        glob.glob(os.path.join(gguf_dir, "*mmproj*.gguf"))
        + glob.glob(os.path.join(gguf_dir, "*vision*.gguf"))
    )
    return candidates[0] if candidates else None


def _ensure_model_ready(gemma4_model):
    """
    确保模型处于可用状态：
    1. 如果 model_obj 为 None（被卸载或缓存复用）→ 重新加载
    2. 如果 GGUF 且 mmproj 状态有变化 → 重新加载
    返回 (True/False, 提示信息)
    """
    import time

    model_obj = gemma4_model.get("model_obj")
    model_format = gemma4_model.get("model_format")
    model_path = gemma4_model.get("model_path")

    need_reload = False
    reason = ""

    if model_obj is None:
        need_reload = True
        reason = "model_obj 为 None（已被卸载或缓存复用）"

    if model_format == "GGUF" and model_obj is not None:
        # 重新扫描当前 mmproj 状态（与加载时记录的 _gemma4_cfg 对比）
        gguf_dir = os.path.dirname(model_path)
        current_mmproj = _detect_mmproj(gguf_dir)
        cfg = getattr(model_obj, "_gemma4_cfg", {})
        old_mmproj = cfg.get("mmproj_path")
        # 判断状态是否改变：之前没有但现在有 / 之前有但现在没有
        if (old_mmproj is None) != (current_mmproj is None):
            need_reload = True
            reason = (
                f"mmproj 状态改变: "
                f"旧={'有:' + os.path.basename(old_mmproj) if old_mmproj else '无'} "
                f"→ 新={'有:' + os.path.basename(current_mmproj) if current_mmproj else '无'}"
            )
        elif old_mmproj and current_mmproj and os.path.abspath(old_mmproj) != os.path.abspath(current_mmproj):
            need_reload = True
            reason = f"mmproj 文件路径改变"

    if not need_reload:
        return True, "模型可用"

    # -------- 执行重新加载 --------
    print(f"[Gemma4Reloader] 触发重新加载: {reason}")

    # 先尝试清理可能残留的对象
    try:
        if model_obj is not None and model_format == "GGUF":
            try:
                model_obj.close()
            except Exception:
                pass
            del model_obj
    except Exception:
        pass

    # 关键修复：llama-cpp-python (llama.cpp) 在 Windows 上 close() 后
    # 底层可能仍持有文件句柄或内存映射。需要等待 OS 释放。
    # 等待时间：2秒 + 递增（最多重试3次）
    max_retries = 3
    base_delay = 2.0  # 秒

    # 从 gemma4_model 中获取原始加载参数（若有保存）
    n_gpu_layers = gemma4_model.get("n_gpu_layers", 0)
    hf_device_map = gemma4_model.get("hf_device_map", "auto")
    hf_dtype = gemma4_model.get("hf_dtype", "bfloat16")
    hf_load_in_4bit = gemma4_model.get("hf_load_in_4bit", False)

    for attempt in range(max_retries):
        try:
            if model_format == "HF":
                new_model_obj, new_processor = _load_hf_model(
                    model_path, device_map=hf_device_map,
                    torch_dtype=hf_dtype, load_in_4bit=hf_load_in_4bit,
                )
            else:
                new_model_obj, new_processor = _load_gguf_model(
                    model_path, n_gpu_layers=n_gpu_layers, n_ctx=8192,
                )
            gemma4_model["model_obj"] = new_model_obj
            gemma4_model["processor"] = new_processor
            print(f"[Gemma4Reloader] ✓ 模型重新加载成功（第 {attempt + 1} 次尝试）")
            return True, "模型已重新加载"
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(f"[Gemma4Reloader] 第 {attempt + 1} 次尝试失败: {last_error[:120]}")
                print(f"[Gemma4Reloader] 等待 {delay:.1f}s 后重试...")
                time.sleep(delay)
            else:
                print(f"[Gemma4Reloader] ✗ 重新加载失败: {last_error}")
                return False, f"重新加载失败: {last_error}"

    # 不应到达这里，但以防万一
    return False, "重新加载失败"


# ============================================================
# 7c. 保存原始加载参数到 gemma4_model（供重新加载使用）
# ============================================================

def _save_loader_params(gemma4_model, **kwargs):
    for k, v in kwargs.items():
        gemma4_model[k] = v


# ============================================================
# 8. 硬编码预设 System Prompt 模板
# ============================================================

from .presets import (
    _EXPAND_PRESETS, _VISION_PRESETS,
    _ALL_PRESET_NAMES, _MERGED_MAP,
)


# ============================================================
# 9. 节点定义
# ============================================================

# ---- 9a. 模型加载器 ----

class Gemma4ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": _model_choices(),
                "n_gpu_layers": ("INT", {
                    "default": 0, "min": -1, "max": 128, "step": 1,
                    "display": "number",
                }),
                "hf_device_map": (
                    ["auto", "cpu", "cuda:0", "cuda:1"], {"default": "auto"},
                ),
                "hf_dtype": (
                    ["bfloat16", "float16", "float32"], {"default": "bfloat16"},
                ),
                "hf_load_in_4bit": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("GEMMA4_MODEL", "STRING",)
    RETURN_NAMES = ("gemma4_model", "model_info",)
    CATEGORY = "AI/Gemma4"
    FUNCTION = "load_model"

    def load_model(self, model_name, n_gpu_layers,
                   hf_device_map, hf_dtype, hf_load_in_4bit):
        full_path, model_format = resolve_model_path(model_name)
        print(f"[Gemma4Loader] 加载: {model_name} ({model_format})")

        if model_format == "HF":
            model_obj, processor = _load_hf_model(
                full_path, device_map=hf_device_map,
                torch_dtype=hf_dtype, load_in_4bit=hf_load_in_4bit,
            )
        else:
            model_obj, processor = _load_gguf_model(
                full_path, n_gpu_layers=n_gpu_layers, n_ctx=8192,
            )

        gemma4_model = {
            "model_obj": model_obj,
            "processor": processor,
            "model_format": model_format,
            "model_path": full_path,
            # 保存原始参数，供重新加载时使用
            "n_gpu_layers": n_gpu_layers,
            "hf_device_map": hf_device_map,
            "hf_dtype": hf_dtype,
            "hf_load_in_4bit": hf_load_in_4bit,
        }

        info_lines = [
            f"模型: {model_name}",
            f"格式: {model_format}",
            f"路径: {full_path}",
        ]
        if model_format == "GGUF":
            cfg = getattr(model_obj, "_gemma4_cfg", {})
            info_lines.append(f"mmproj: {'YES' if cfg.get('is_mm') else 'NO (纯文本模式)'}")
            info_lines.append(f"GPU layers: {n_gpu_layers}")
        else:
            info_lines.append(f"device_map: {hf_device_map}")
            info_lines.append(f"dtype: {hf_dtype}")
            info_lines.append(f"4bit: {'YES' if hf_load_in_4bit else 'NO'}")

        print(f"[Gemma4Loader] ✓ 加载完成")
        return (gemma4_model, "\n".join(info_lines))


# ---- 9b. 主节点（图像反推 + 提示词扩写） ----

class Gemma4ImageCaption:
    """
    双模式节点:
      接图像 → 图像反推（多模态）
      不接图 → 提示词扩写优化（纯文本）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemma4_model": ("GEMMA4_MODEL",),
                "system_preset": (_ALL_PRESET_NAMES, {"default": _ALL_PRESET_NAMES[0]}),
                "max_length": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "temperature": (
                    "FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
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
                    "placeholder": "User Prompt（可选，如要扩写的提示词）",
                }),
                "image": ("IMAGE",),
                "auto_unload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    CATEGORY = "AI/Gemma4"
    FUNCTION = "process"

    def process(self, gemma4_model, system_preset, max_length, temperature,
                system_prompt="", user_prompt="", image=None, auto_unload=True):
        # 关键修复：确保模型可用（处理 None + mmproj 状态变更）
        ok, info = _ensure_model_ready(gemma4_model)
        if not ok:
            return (f"[错误] 模型无法使用: {info}",)

        has_image = _has_valid_image(image)

        final_system = (
            system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else _MERGED_MAP.get(system_preset, _MERGED_MAP.get(_ALL_PRESET_NAMES[0], ""))
        )
        final_user = user_prompt.strip() if user_prompt else ""

        if has_image:
            pil_image = _image_to_pil(image)
            result = run_inference(
                gemma4_model["model_obj"],
                gemma4_model["processor"],
                gemma4_model["model_format"],
                pil_image,
                system_prompt=final_system,
                user_prompt=final_user,
                max_new_tokens=max_length,
                temperature=temperature,
            )
        else:
            result = run_text_inference(
                gemma4_model["model_obj"],
                gemma4_model["processor"],
                gemma4_model["model_format"],
                system_prompt=final_system,
                user_prompt=final_user,
                max_new_tokens=max_length,
                temperature=temperature,
            )

        if auto_unload:
            _unload_gemma4_model(gemma4_model)

        return (result,)


# ---- 9c. 批量描述节点 ----

class Gemma4BatchCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemma4_model": ("GEMMA4_MODEL",),
                "images": ("IMAGE",),
                "system_preset": (_ALL_PRESET_NAMES, {"default": _ALL_PRESET_NAMES[0]}),
                "max_length": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 32}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "自定义 System Prompt（留空则使用上方预设）",
                }),
                "user_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "User Prompt（可选）",
                }),
                "auto_unload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("captions",)
    CATEGORY = "AI/Gemma4"
    FUNCTION = "batch_caption"

    def batch_caption(self, gemma4_model, images, system_preset, max_length,
                      system_prompt="", user_prompt="", auto_unload=True):
        # 关键修复：确保模型可用（处理 None + mmproj 状态变更）
        ok, info = _ensure_model_ready(gemma4_model)
        if not ok:
            return (f"[错误] 模型无法使用: {info}",)

        final_system = (
            system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else _MERGED_MAP.get(system_preset, _ALL_PRESET_NAMES[0])
        )
        final_user = user_prompt.strip() if user_prompt else ""

        captions = []
        for i in range(images.shape[0]):
            arr = (images[i].cpu().numpy() * 255).astype("uint8")
            pil_image = Image.fromarray(arr)
            caption = run_inference(
                gemma4_model["model_obj"],
                gemma4_model["processor"],
                gemma4_model["model_format"],
                pil_image,
                system_prompt=final_system,
                user_prompt=final_user,
                max_new_tokens=max_length, temperature=0.7,
            )
            captions.append(caption)

        if auto_unload:
            _unload_gemma4_model(gemma4_model)
        return ("\n".join(captions),)


# ---- 9d. 模型列表信息 ----

class Gemma4ModelInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("models_list", "model_directory", "presets_info")
    CATEGORY = "AI/Gemma4"
    FUNCTION = "list_models"

    def list_models(self):
        gemma4_root = os.path.join(folder_paths.models_dir, GEMMA4_MODEL_DIR)
        models = list_gemma4_models()
        lines = []
        for m in models:
            try:
                path, fmt = resolve_model_path(m)
                if fmt == "GGUF":
                    directory = os.path.dirname(path)
                    mm = (
                        glob.glob(os.path.join(directory, "*mmproj*.gguf"))
                        + glob.glob(os.path.join(directory, "*vision*.gguf"))
                    )
                    mm_status = (
                        os.path.basename(mm[0]) if mm
                        else "(未找到 — 将以纯文本 LLM 运行)"
                    )
                    lines.append(f"{m}\n  路径: {path}\n  mmproj: {mm_status}")
                else:
                    lines.append(f"{m}\n  路径: {path}")
            except Exception as e:
                lines.append(f"{m}\n  [解析失败] {e}")
        if not lines:
            lines.append("(未发现任何模型)")
        lines.append(f"\n模型目录: {gemma4_root}")

        presets_info = (
            f"扩写预设 ({len(_EXPAND_PRESETS)}): "
            + ", ".join(l for l, _ in _EXPAND_PRESETS)
            + f"\n\n视觉预设 ({len(_VISION_PRESETS)}): "
            + ", ".join(l for l, _ in _VISION_PRESETS)
        )

        return ("\n\n".join(lines), gemma4_root, presets_info)


# ---- 9e. 模型卸载 ----

def _unload_gemma4_model(gemma4_model):
    import gc
    model_obj = gemma4_model.get("model_obj")
    model_format = gemma4_model.get("model_format", "unknown")
    if model_obj is None:
        print(f"[Gemma4Unloader] 模型已为 None，跳过卸载")
        return
    print(f"[Gemma4Unloader] 卸载 {model_format} 模型...")
    try:
        if model_format == "HF":
            if hasattr(model_obj, "to"):
                model_obj.to("cpu")
            # 尝试释放 HuggingFace 模型内部张量
            del model_obj
        elif model_format == "GGUF":
            #  llama-cpp-python 需要明确 close()
            try:
                model_obj.close()
            except Exception as e1:
                print(f"[Gemma4Unloader]   close() 失败 (可忽略): {e1}")
            try:
                # 尝试调用 free / _free_ctx 等内部方法
                if hasattr(model_obj, "free"):
                    model_obj.free()
                elif hasattr(model_obj, "_free_ctx"):
                    model_obj._free_ctx()
            except Exception as e2:
                print(f"[Gemma4Unloader]   free 调用失败 (可忽略): {e2}")
            # 销毁实例，触发 __del__
            del model_obj
    except Exception as e:
        print(f"[Gemma4Unloader]   卸载过程异常 (可忽略): {e}")

    # 用 None 标记"已卸载"，但注意不要破坏被 ComfyUI 缓存的对象
    # 恢复时将使用 reload 机制重新加载
    gemma4_model["model_obj"] = None
    gemma4_model["processor"] = None
    # 强制 gc 收集 + CUDA 同步
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        # 额外调用一次释放 IPU 资源
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print(f"[Gemma4Unloader] ✓ 卸载完成")


class Gemma4ModelUnloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"gemma4_model": ("GEMMA4_MODEL",)}}

    RETURN_TYPES = ()
    CATEGORY = "AI/Gemma4"
    FUNCTION = "unload"
    OUTPUT_NODE = True

    def unload(self, gemma4_model):
        _unload_gemma4_model(gemma4_model)
        return ()


# ============================================================
# 10. 节点注册
# ============================================================

# 导入 API 节点模块
from .api_node import RhLlmApiNode

NODE_CLASS_MAPPINGS = {
    "Gemma4ModelLoader":   Gemma4ModelLoader,
    "Gemma4ModelUnloader": Gemma4ModelUnloader,
    "Gemma4ImageCaption":  Gemma4ImageCaption,
    "Gemma4BatchCaption":  Gemma4BatchCaption,
    "Gemma4ModelInfo":     Gemma4ModelInfo,
    "RhLlmApiNode":        RhLlmApiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemma4ModelLoader":   "Gemma4 Model Loader     [加载模型]",
    "Gemma4ModelUnloader": "Gemma4 Model Unloader    [卸载/释放显存]",
    "Gemma4ImageCaption":  "Gemma4 Process           [图像反推⊕提示词扩写]",
    "Gemma4BatchCaption":  "Gemma4 Batch Caption     [批量描述]",
    "Gemma4ModelInfo":     "Gemma4 Model Info         [模型列表]",
    "RhLlmApiNode":        "RH LLM API Node         [API调用:图/视频/文本]",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
