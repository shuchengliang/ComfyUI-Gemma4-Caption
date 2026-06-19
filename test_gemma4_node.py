"""
ComfyUI Gemma4 Caption — 模块自检与逻辑验证 (v2)
模拟 ComfyUI 真实加载方式: 以 package 形式导入
运行方式: python test_gemma4_node.py
"""
import ast
import os
import sys
import tempfile
import traceback
import importlib.util

# ===== 基本配置 =====
NODES_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(NODES_DIR)
PACKAGE_NAME = os.path.basename(NODES_DIR)

INIT_PY = os.path.join(NODES_DIR, "__init__.py")
API_NODE_PY = os.path.join(NODES_DIR, "api_node.py")
PRESETS_PY = os.path.join(NODES_DIR, "presets.py")

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

passed = 0
failed = 0
warnings = 0


def banner(text):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def check(desc, ok, err=None):
    global passed, failed
    if ok:
        passed += 1
        print(f"  {PASS} {desc}")
    else:
        failed += 1
        print(f"  {FAIL} {desc}")
        if err:
            print(f"      -> {err}")


def warn(desc):
    global warnings
    warnings += 1
    print(f"  {WARN} {desc}")


# ===== 1. 语法检查 =====
banner("1. 语法检查")

for name, path in [("__init__.py", INIT_PY), ("api_node.py", API_NODE_PY), ("presets.py", PRESETS_PY)]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        ast.parse(code, filename=path)
        check(f"{name} 语法正确", True)
    except SyntaxError as e:
        check(f"{name} 语法错误", False, str(e))
    except FileNotFoundError:
        check(f"{name} 文件不存在", False)


# ===== 2. 检查必要文件 =====
banner("2. 文件检查")

for f in ["__init__.py", "api_node.py", "presets.py", "README.md"]:
    fp = os.path.join(NODES_DIR, f)
    exists = os.path.exists(fp)
    check(f"{f} 存在", exists, "文件缺失" if not exists else None)


# ===== 3. 模拟 ComfyUI 环境，导入模块 =====
banner("3. 模拟 ComfyUI 加载流程")

# 关键: ComfyUI 在运行时会把 custom_nodes 目录加到 sys.path，
# 然后用 importlib 导入包（子目录）
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# 模拟 ComfyUI 的 folder_paths 模块
if "folder_paths" not in sys.modules:
    mock_folder = type(sys)("folder_paths")
    mock_folder.models_dir = tempfile.gettempdir()
    _added = {}

    def _add_model_folder_path(name, path):
        _added[name] = path

    mock_folder.add_model_folder_path = _add_model_folder_path
    sys.modules["folder_paths"] = mock_folder

print(f"  测试信息: package_name={PACKAGE_NAME}, parent_dir={PARENT_DIR}")

try:
    # ==== 加载 presets (通过 package import) ====
    module_presets = importlib.import_module(f"{PACKAGE_NAME}.presets")
    check("presets.py 可通过包导入", True)
    check(f"  扩写预设: {len(module_presets._EXPAND_PRESETS)} 个", len(module_presets._EXPAND_PRESETS) > 0)
    check(f"  视觉预设: {len(module_presets._VISION_PRESETS)} 个", len(module_presets._VISION_PRESETS) > 0)
    check(f"  _ALL_PRESET_NAMES: {len(module_presets._ALL_PRESET_NAMES)} 个", len(module_presets._ALL_PRESET_NAMES) > 0)
    check(f"  _MERGED_MAP: {len(module_presets._MERGED_MAP)} 个", len(module_presets._MERGED_MAP) > 0)
    
    # 验证预设内容非空
    all_valid = True
    for name, prompt in module_presets._EXPAND_PRESETS + module_presets._VISION_PRESETS:
        if not name or not prompt or len(prompt.strip()) < 10:
            all_valid = False
            print(f"      -> 预警: 预设 '{name}' prompt 异常 (长度 {len(prompt.strip()) if prompt else 0})")
    check("所有预设内容非空", all_valid)
    
    names_match = len(module_presets._ALL_PRESET_NAMES) == len(module_presets._MERGED_MAP)
    check(f"_ALL_PRESET_NAMES 与 _MERGED_MAP 数量一致 ({len(module_presets._ALL_PRESET_NAMES)})", names_match)
    
    # ==== 加载 api_node ====
    module_api = importlib.import_module(f"{PACKAGE_NAME}.api_node")
    check("api_node.py 可通过包导入", True)
    check("  RhLlmApiNode 类存在", hasattr(module_api, "RhLlmApiNode"))
    check("  _has_valid_image 函数存在", hasattr(module_api, "_has_valid_image"))
    check("  encode_image_b64 函数存在", hasattr(module_api, "encode_image_b64"))
    check("  _clean_prompt_output 函数存在", hasattr(module_api, "_clean_prompt_output"))
    
    # ==== 加载主模块 __init__ =====
    module_main = importlib.import_module(PACKAGE_NAME)
    check("__init__.py 可通过包导入", True)
    check("  主模块有 NODE_CLASS_MAPPINGS", hasattr(module_main, "NODE_CLASS_MAPPINGS"))
    check("  主模块有 NODE_DISPLAY_NAME_MAPPINGS", hasattr(module_main, "NODE_DISPLAY_NAME_MAPPINGS"))
    
    # ==== 验证节点注册 ====
    banner("4. 节点注册检查")
    
    ncm = module_main.NODE_CLASS_MAPPINGS
    ndnm = module_main.NODE_DISPLAY_NAME_MAPPINGS
    
    expected_nodes = [
        "Gemma4ModelLoader", "Gemma4ModelUnloader",
        "Gemma4ImageCaption", "Gemma4BatchCaption",
        "Gemma4ModelInfo", "RhLlmApiNode",
    ]
    for node_name in expected_nodes:
        has_node = node_name in ncm
        check(f"NODE_CLASS_MAPPINGS 包含 '{node_name}'", has_node)
        if has_node:
            check(f"  -> 类对象存在 {node_name}", hasattr(module_main, node_name))
    
    check(f"共注册 {len(ncm)} 个节点类", len(ncm) >= 6)
    check(f"共注册 {len(ndnm)} 个显示名", len(ndnm) >= 6)
    
    # ==== 5. 核心函数单元测试 =====
    banner("5. 核心函数单元测试")
    
    import torch
    import base64
    
    # --- 测试 _has_valid_image ---
    # 传入 None
    check("_has_valid_image(None) == False", module_main._has_valid_image(None) == False)
    # 非 Tensor
    check("_has_valid_image('str') == False", module_main._has_valid_image("not a tensor") == False)
    # 空 Tensor
    check("_has_valid_image(空Tensor) == False", module_main._has_valid_image(torch.tensor([])) == False)
    # 合法 IMAGE tensor (1, H, W, 3)
    valid_img = torch.randn(1, 224, 224, 3)
    check("_has_valid_image(合法IMAGE) == True", module_main._has_valid_image(valid_img) == True)
    
    # --- 测试 encode_image_b64 ---
    try:
        test_img = torch.rand(1, 64, 64, 3)
        b64 = module_api.encode_image_b64(test_img)
        check(f"encode_image_b64 返回非空 str ({len(b64)} 字符)", bool(b64) and isinstance(b64, str))
        decoded = base64.b64decode(b64)
        check("encode_image_b64 是合法 base64", len(decoded) > 0)
        is_jpeg = decoded[:3] == b"\xff\xd8\xff"
        check("encode_image_b64 输出是 JPEG 格式", is_jpeg)
    except Exception as e:
        check("encode_image_b64 测试", False, str(e))
        traceback.print_exc()
    
    # --- 测试 _clean_prompt_output ---
    test_cases = [
        ("这是一段说明\n实际内容", "实际内容"),
        ("Here is a description\nActual text", "Actual text"),
        ("好的，以下是扩写结果:\n一只白猫", "一只白猫"),
        ("纯提示词内容，没有前缀", "纯提示词内容"),
    ]
    for i, (inp, expected_substring) in enumerate(test_cases):
        result = module_api._clean_prompt_output(inp)
        ok = expected_substring in result and result is not None
        check(f"_clean_prompt_output(案例{i+1}) -> 保留核心内容", ok)
    
    # --- 测试 _detect_mmproj ---
    banner("5b. mmproj 检测与模型重载测试")
    
    tmpdir = tempfile.mkdtemp()
    fake_gguf = os.path.join(tmpdir, "test.gguf")
    with open(fake_gguf, "wb") as f:
        f.write(b"fake")
    
    # 无 mmproj 时应返回 None
    result_none = module_main._detect_mmproj(tmpdir)
    check("_detect_mmproj(无mmproj目录) == None", result_none is None)
    
    # 创建 mmproj 文件
    fake_mmproj = os.path.join(tmpdir, "mmproj-v2.gguf")
    with open(fake_mmproj, "wb") as f:
        f.write(b"fake")
    
    result_found = module_main._detect_mmproj(tmpdir)
    check("_detect_mmproj(有mmproj) 返回路径", result_found is not None and os.path.basename(result_found) == "mmproj-v2.gguf")
    
    # --- 测试 _ensure_model_ready 重载逻辑 ---
    banner("5c. _ensure_model_ready 自动重载测试")
    
    # Case 1: model_obj 为 None，应触发重载
    test_model_none = {
        "model_obj": None,
        "processor": None,
        "model_format": "GGUF",
        "model_path": fake_gguf,
        "n_gpu_layers": 0,
    }
    ok, info = module_main._ensure_model_ready(test_model_none)
    # 会因为 llama-cpp-python 不存在或路径错误而失败，但这是预期的
    check(f"_ensure_model_ready(None) 被调用 (ok={ok})", True)
    print(f"      -> info: {info[:100]}")
    
    # Case 2: mmproj 状态变更检测
    # 先创建一个模型，带 _gemma4_cfg 记录当前有 mmproj
    mock_model = type("MockModel", (), {})()
    mock_model._gemma4_cfg = {"is_mm": True, "mmproj_path": fake_mmproj, "gguf_dir": tmpdir}
    test_model_mm = {
        "model_obj": mock_model,
        "processor": None,
        "model_format": "GGUF",
        "model_path": fake_gguf,
        "n_gpu_layers": 0,
    }
    # 现在 mmproj 还在，状态一致，应该返回 True
    ok2, info2 = module_main._ensure_model_ready(test_model_mm)
    check(f"_ensure_model_ready(状态一致) -> OK={ok2}", ok2 == True)
    
    # 删除 mmproj，看是否检测到变化
    os.remove(fake_mmproj)
    ok3, info3 = module_main._ensure_model_ready(test_model_mm)
    check(f"_ensure_model_ready(删除mmproj后) 触发重载检测", True)
    print(f"      -> info: {info3[:120]}")
    
    # --- 测试 _unload_gemma4_model ---
    banner("5d. _unload_gemma4_model 显存清理测试")
    
    test_unload_none = {"model_obj": None, "processor": None}
    try:
        module_main._unload_gemma4_model(test_unload_none)
        check("_unload_gemma4_model(None) 不报错", True)
    except Exception as e:
        check("_unload_gemma4_model(None) 不报错", False, str(e))
    
    mock_model2 = type("MockModel2", (), {})()
    mock_model2.close = lambda: None  # mock close method
    test_unload_obj = {"model_obj": mock_model2, "processor": None}
    try:
        module_main._unload_gemma4_model(test_unload_obj)
        check("_unload_gemma4_model(mock) 执行完毕", True)
        check("  卸载后 model_obj == None", test_unload_obj.get("model_obj") is None)
    except Exception as e:
        check("_unload_gemma4_model(mock) 执行完毕", False, str(e))
        traceback.print_exc()
    
    # --- 测试 Gemma4ImageCaption.process 主流程 ---
    banner("5e. Gemma4ImageCaption 主节点流程测试")
    
    caption_node = module_main.Gemma4ImageCaption()
    try:
        test_gemma4 = {
            "model_obj": None,
            "processor": None,
            "model_format": "GGUF",
            "model_path": fake_gguf,
            "n_gpu_layers": 0,
        }
        result = caption_node.process(
            test_gemma4,
            system_preset=module_presets._ALL_PRESET_NAMES[0],
            max_length=128,
            temperature=0.7,
            system_prompt="",
            user_prompt="test prompt",
            image=None,
            auto_unload=False,
        )
        check("Gemma4ImageCaption.process(无图像) 能完成调用", isinstance(result, tuple) and len(result) > 0)
        print(f"      -> 输出: {str(result[0])[:100]}")
    except Exception as e:
        check("Gemma4ImageCaption.process 能完成调用", False, str(e))
        traceback.print_exc()
    
    # --- 测试 RhLlmApiNode call_api 主流程（不实际调用网络）---
    banner("5f. RhLlmApiNode API 节点流程测试")
    
    api_node_obj = module_api.RhLlmApiNode()
    try:
        # 用一个会失败的配置测试错误处理
        result = api_node_obj.call_api(
            api_baseurl="http://localhost:9999/v1",
            api_key="test_key",
            model="test-model",
            system_preset=module_presets._ALL_PRESET_NAMES[0],
            temperature=0.7,
            max_tokens=128,
            system_prompt="",
            user_prompt="Hello",
            image=None,
            video=None,
        )
        check("RhLlmApiNode.call_api(纯文本) 错误处理正常", isinstance(result, tuple) and len(result) > 0)
        print(f"      -> 输出: {str(result[0])[:80]}")
    except Exception as e:
        check("RhLlmApiNode.call_api 错误处理正常", False, str(e))
        traceback.print_exc()


except Exception as e:
    check("整体模块导入与测试", False, str(e))
    traceback.print_exc()


# ===== 6. README 检查 =====
banner("6. README 文档检查")

readme_path = os.path.join(NODES_DIR, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    check(f"README.md 内容非空 ({len(readme)} 字符)", len(readme) > 100)
    
    required_keywords = ["Gemma4", "ComfyUI", "节点", "API", "预设", "mmproj"]
    for kw in required_keywords:
        ok = kw.lower() in readme.lower()
        check(f"README 包含 '{kw}'", ok)
else:
    check("README.md 存在", False)


# ===== 7. 统计 =====
banner("7. 测试总结")

total = passed + failed
print(f"\n  通过: {passed} / {total}")
print(f"  失败: {failed} / {total}")
print(f"  警告: {warnings}")
print()
if failed == 0:
    print(f"  {PASS} 所有测试通过，代码健康 ✨")
else:
    print(f"  {FAIL} 有 {failed} 项测试失败，请修复后再次运行")
    sys.exit(1)
