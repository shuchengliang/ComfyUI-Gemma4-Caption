# ComfyUI Gemma4 12B — 图像反推 & 提示词扩写 & API 调用

> **版本**: 1.1
> **本地模型**: Google Gemma 4 12B (GGUF / HuggingFace 双格式)
> **API 支持**: OpenAI 兼容接口（DeepSeek、Qwen、通义千问等）
> **功能**: 图像反推、提示词撰写/扩写优化、API 多模态调用
> **预设模板来源**: 提示词小助手

---

## ✨ 功能概述

一个 ComfyUI 自定义节点插件，支持两大推理方式：

| 模式 | 说明 |
|------|------|
| **本地模型** | 使用 Gemma4 12B GGUF/HF 模型，完全离线运行 |
| **API 调用** | 接入 OpenAI 兼容接口，支持图像、视频、纯文本，支持所有 Gemma4 预设模板 |

**双模式节点设计**：本地模型和 API 节点共享同一套预设模板体系。

## 🏗️ 架构设计

### 本地模型（AI/Gemma4）

```
Gemma4ModelLoader ──(GEMMA4_MODEL)──> Gemma4ImageCaption   (主节点：反推/扩写 双模式)
                    ──(GEMMA4_MODEL)──> Gemma4BatchCaption     (批量图像描述)
                    ──(GEMMA_FILE)──> Gemma4ModelUnloader      (手动释放显存)
```

### API 调用（AI/LLM API）

```
RhLlmApiNode ──(output)──> 下游节点
  ├─ 有图: 图像理解（base64 JPEG → vision API）
  ├─ 有视频: 视频理解（base64 MP4 → video API）
  └─ 无图/视频: 纯文本对话
```

- **一份模型，多处共享** — Loader 加载后输出连线，所有下游节点共享同一份模型
- **预设模板统一** — 本地节点和 API 节点共享 14 个预设 System Prompt
- **管道式设计** — 符合 ComfyUI 标准工作流模式

## 📦 支持的模型格式

### 本地模型

| 格式 | 加载方式 | 视觉支持 | 推荐用途 |
|------|---------|---------|---------|
| **GGUF** | llama-cpp-python + Gemma4ChatHandler | ✅ 需要 mmproj 视觉投影 | 量化模型、低显存（~4GB） |
| **HuggingFace** | transformers + AutoModelForImageTextToText | ✅ 原生支持 | 标准精度（~12GB bf16） |

> GGUF 模型必须同时放置 mmproj 视觉投影文件（`*mmproj*.gguf`），否则只能纯文本推理。

### API 接口

| 接口类型 | 示例模型 | 支持图像 | 支持视频 |
|---------|---------|---------|---------|
| OpenAI 兼容 | GPT-4o, GPT-4 Vision | ✅ | ✅ (部分) |
| DeepSeek | deepseek-chat, deepseek-vl | ✅ | ✅ |
| 通义千问 | qwen-vl-max, qwen2.5-omni | ✅ | ✅ |
| 自建接口 | Anything with OpenAI-compatible endpoint | ✅ | ✅ |

## 🎛️ 节点一览

### 本地模型节点（AI/Gemma4）

| 节点名 | 输入 | 输出 | 说明 |
|--------|------|------|------|
| **Gemma4 Model Loader** | model_name, n_gpu_layers… | `GEMMA4_MODEL`, `model_info` | 加载并缓存本地模型 |
| **Gemma4 Process** | `GEMMA4_MODEL`, system_preset, user_prompt, image(可选) | `output` | 🎯 主节点：图像反推 ⊕ 提示词扩写 |
| **Gemma4 Batch Caption** | `GEMMA4_MODEL`, images | `captions` | 批量图像逐张描述 |
| **Gemma4 Model Unloader** | `GEMMA4_MODEL` | 无 | 主动释放 GPU 显存 |
| **Gemma4 Model Info** | 无 | `models_list`, `model_directory`, `presets_info` | 诊断：列出模型 & mmproj 状态 & 预设列表 |

### API 节点（AI/LLM API）

| 节点名 | 输入 | 输出 | 说明 |
|--------|------|------|------|
| **RH LLM API Node** | api_baseurl, api_key, model, system_preset, image/video(可选) | `output` | OpenAI 兼容接口，支持图像/视频/文本 |

## 📝 预设 System Prompt 模板

**预设模板来源于「提示词小助手」项目的 `system_prompts_template.json`，经授权后硬编码嵌入。**

下拉框包含 **14 个专业预设**，**本地模型和 API 节点均可使用**：

### 扩写类 (7 个)

| 预设名 | 用途 |
|--------|------|
| 扩写-通用 | 全领域自适应扩写（摄影/工业/动漫/3D…） |
| 扩写-人像大师 | 人像摄影提示词，五维细节填充 |
| 扩写-Tags风格 | Danbooru 标签 + SD 权重语法 |
| Qwen-Image-Edit指令优化 | Qwen 图像编辑指令生成 |
| Kontext指令优化并翻译 | Flux.1 Kontext 编辑指令（英文） |
| Wan视频提示词 | 通义万相视频提示词（主体+场景+运动+美学+风格） |

### 视觉/反推类 (7 个)

| 预设名 | 用途 |
|--------|------|
| 像素级描述 | 全要素像素级图片反推 |
| 图像描述-Tag风格 | SD/Flux 标签化反推 |
| 图像编辑重绘 | 人像编辑指令生成 |
| Qwen-Image-Edit指令优化-视觉版 | Qwen 编辑指令（视觉版） |
| i2v物理动态提示词 | 图像→视频 动态提示词 |
| Kontext指令优化并翻译-视觉版 | Kontext 视觉编辑指令 |
| Detail Caption | 全英文像素级反推 |
| Caption-Tags | 全英文 Tags 反推 |

**System Prompt 优先级规则**：`自定义 system_prompt` > `预设下拉选择` > `default`

## 🚀 快速开始

### 1. 安装

```bash
# 将插件文件夹放入 ComfyUI custom_nodes 目录
cd ComfyUI/custom_nodes
git clone https://github.com/shuchengliang/ComfyUI-Gemma4-Caption.git

# 安装依赖
pip install torch>=2.0.0 Pillow>=9.0.0 transformers>=4.36.0 openai>=1.0.0

# GGUF 格式支持（需 GPU 版本参考下方）
pip install llama-cpp-python
```

### 2. 使用

#### 方式一：API 调用（推荐，无 GPU 也能用）

1. 打开 ComfyUI → **AI/LLM API** → 添加 **RH LLM API Node**
2. 填写 `api_baseurl`（如 `https://api.openai.com/v1`）
3. 填写 `api_key`（支持环境变量 `OPENAI_API_KEY`）
4. 选择 `model`（如 `gpt-4o`、`deepseek-chat`）
5. 选择预设模板、填入 user_prompt（可选）、接入图像（可选）
6. 运行

#### 方式二：本地模型

1. 打开 ComfyUI → **AI/Gemma4** → 添加 **Gemma4 Model Loader**
2. 选择模型、GPU offload 层数 → 运行加载
3. 添加 **Gemma4 Process** 节点
4. 选择预设模板、填入 user_prompt（可选）、接入图像（可选）
5. 运行

### 3. 准备本地模型（仅本地模式需要）

将模型文件放入 ComfyUI 标准模型目录：

```
ComfyUI/models/gemma4/
├── gemma-4-12b-it/                       # HF 格式（目录 + config.json + safetensors）
├── gemma-4-12b-it-qat-q4_0.gguf         # GGUF 主模型
└── mmproj-gemma-4-12b-it-qat-q4_0.gguf  # GGUF 视觉投影（必需！）
```

## 🔧 llama-cpp-python GPU 加速

```powershell
# CUDA 13.2（RTX 50 系列）
pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.28-cu132/llama_cpp_python-0.3.28-py3-none-win_amd64.whl"
```

其他 CUDA 版本参考 [abetlen/llama-cpp-python Releases](https://github.com/abetlen/llama-cpp-python/releases)。

## 📁 项目结构

```
ComfyUI_Gemma4_Caption/
├── __init__.py    # 主代码：Gemma4 本地模型加载、推理、节点定义
├── api_node.py    # API 调用节点：OpenAI 兼容接口，支持图像/视频
├── presets.py     # 预设 System Prompt 模板（硬编码，无需外部文件）
├── README.md      # 本文档
└── .gitignore
```

## ⚠️ 注意事项

1. **API Node 依赖** — 需要 `pip install openai`
2. **GGUF 必须配 mmproj** — 否则图像反推会返回 "缺少视觉投影" 提示
3. **显存管理** — 默认反推完成后自动释放显存（auto_unload），可关闭
4. **首次加载慢** — 12B 模型首次加载需 30s~数分钟，后续节点复用缓存
5. **输出清洗** — 代码内置 `_clean_prompt_output()` 自动去除模型输出的开场白废话
6. **模型自动重载** — 运行一次后更换预设或添加 mmproj 文件，节点会自动重新加载模型
7. **文件句柄等待** — Windows 上 llama-cpp-python 重新加载时会等待最多 6 秒（重试 3 次）

## 📄 致谢

- **提示词小助手** — 提供 `system_prompts_template.json` 预设模板体系
- **ComfyUI_RH_LLM_API** — 提供 OpenAI 兼容 API 调用架构参考
- [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — GGUF Python 绑定
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — 强大的 AI 图像生成工作流引擎
- Google Gemma 4 Team

## 📜 License

MIT
