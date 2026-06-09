#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化 DPS / score model 加载检查脚本

目标：
1. 导入 score model 配置
2. 构建 score model
3. 尝试加载 checkpoint
4. 检查 checkpoint 是否存在、是否匹配
5. 不做采样，只做一次前向接口检查

推荐运行方式：
python test_dps_model_loading.py \
  --repo_root /path/to/USCT_download \
  --checkpoint /path/to/checkpoint.pth \
  --device cuda:0

如果暂时没有 checkpoint，也可以先只检查模型构建：
python test_dps_model_loading.py --repo_root /path/to/USCT_download
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_repo_on_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def check_basic_dependencies() -> None:
    required = ["torch", "ml_collections", "numpy"]
    missing = []
    for name in required:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "缺少依赖包: {}\n"
            "请先安装，例如:\n"
            "  pip install {}".format(
                ", ".join(missing),
                " ".join(missing),
            )
        )


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_config(device, use_dataparallel: bool):
    from configs.vp import AI4Scup2_ddpm_continuous as configs

    config = configs.get_config()
    if device.type == "cuda":
        gpu_index = 0 if device.index is None else device.index
        config.device_ids = [gpu_index] if not use_dataparallel else [gpu_index]
    else:
        config.device_ids = []
    config.device = device
    return config


def build_model(config, device, use_dataparallel: bool):
    import torch
    from models import ddpm as ddpm_model

    base_model = ddpm_model.DDPM(config)
    base_model = base_model.to(device)

    if device.type == "cuda" and use_dataparallel and len(config.device_ids) > 0:
        model = torch.nn.DataParallel(base_model, device_ids=config.device_ids)
    else:
        model = base_model

    model.eval()
    return model, base_model


def unwrap_state_dict(ckpt: Any) -> Tuple[Dict[str, Any], str]:
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "ema", "ema_state_dict", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
        # 有些 checkpoint 直接就是参数字典
        if all(isinstance(v, (int, float, str, bytes, dict, list, tuple)) or hasattr(v, "shape") for v in ckpt.values()):
            tensor_like_keys = [k for k, v in ckpt.items() if hasattr(v, "shape")]
            if tensor_like_keys:
                return ckpt, "<root>"
    raise RuntimeError("无法从 checkpoint 中识别 state_dict 结构。")


def strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def add_prefix_if_needed(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(not k.startswith(prefix) for k in keys):
        return {prefix + k: v for k, v in state_dict.items()}
    return state_dict


def adapt_state_dict_for_model(model, state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    model_has_module = model_keys[0].startswith("module.") if model_keys else False
    ckpt_has_module = ckpt_keys[0].startswith("module.") if ckpt_keys else False

    note = []
    adapted = state_dict
    if model_has_module and not ckpt_has_module:
        adapted = add_prefix_if_needed(adapted, "module.")
        note.append("给 checkpoint keys 自动加上 'module.' 前缀")
    elif ckpt_has_module and not model_has_module:
        adapted = strip_prefix_if_present(adapted, "module.")
        note.append("把 checkpoint keys 的 'module.' 前缀去掉")
    else:
        note.append("keys 前缀无需调整")

    return adapted, "; ".join(note)


def summarize_key_match(model, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    ckpt_keys = set(state_dict.keys())

    common_keys = sorted(model_keys & ckpt_keys)
    missing_keys = sorted(model_keys - ckpt_keys)
    unexpected_keys = sorted(ckpt_keys - model_keys)

    shape_mismatch = []
    for k in common_keys:
        if tuple(model_sd[k].shape) != tuple(state_dict[k].shape):
            shape_mismatch.append((k, tuple(model_sd[k].shape), tuple(state_dict[k].shape)))

    exact_match_keys = [k for k in common_keys if tuple(model_sd[k].shape) == tuple(state_dict[k].shape)]

    return {
        "num_model_keys": len(model_keys),
        "num_ckpt_keys": len(ckpt_keys),
        "num_common_keys": len(common_keys),
        "num_exact_match_keys": len(exact_match_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch": shape_mismatch,
    }


def load_checkpoint(model, checkpoint_path: Path, device, strict: bool) -> None:
    import torch

    print_header("[3] 检查并加载 checkpoint")
    if not checkpoint_path.exists():
        print(f"[FAIL] checkpoint 不存在: {checkpoint_path}")
        return

    print(f"[OK] checkpoint 存在: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"checkpoint 顶层类型: {type(ckpt).__name__}")
    if isinstance(ckpt, dict):
        print(f"checkpoint 顶层 keys: {list(ckpt.keys())[:20]}")

    state_dict, source_key = unwrap_state_dict(ckpt)
    print(f"识别到 state_dict 来源: {source_key}")

    adapted_sd, note = adapt_state_dict_for_model(model, state_dict)
    print(f"keys 适配说明: {note}")

    summary = summarize_key_match(model, adapted_sd)
    print(f"model 参数键数      : {summary['num_model_keys']}")
    print(f"checkpoint 参数键数 : {summary['num_ckpt_keys']}")
    print(f"共同参数键数        : {summary['num_common_keys']}")
    print(f"形状完全匹配键数    : {summary['num_exact_match_keys']}")
    print(f"missing keys 数量    : {len(summary['missing_keys'])}")
    print(f"unexpected keys 数量 : {len(summary['unexpected_keys'])}")
    print(f"shape mismatch 数量  : {len(summary['shape_mismatch'])}")

    if summary["missing_keys"]:
        print("\n前 20 个 missing keys:")
        for k in summary["missing_keys"][:20]:
            print("  -", k)

    if summary["unexpected_keys"]:
        print("\n前 20 个 unexpected keys:")
        for k in summary["unexpected_keys"][:20]:
            print("  -", k)

    if summary["shape_mismatch"]:
        print("\n前 20 个 shape mismatch:")
        for k, ms, cs in summary["shape_mismatch"][:20]:
            print(f"  - {k}: model={ms}, ckpt={cs}")

    # 只有形状匹配的键才真正加载，避免直接报错
    filtered_sd = {}
    model_sd = model.state_dict()
    for k, v in adapted_sd.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            filtered_sd[k] = v

    print(f"\n实际准备加载的参数键数: {len(filtered_sd)}")
    missing_after_filter = sorted(set(model_sd.keys()) - set(filtered_sd.keys()))
    unexpected_after_filter = sorted(set(filtered_sd.keys()) - set(model_sd.keys()))

    result = model.load_state_dict(filtered_sd, strict=False if not strict else strict)
    print("\nload_state_dict 返回:")
    print("  missing_keys   =", len(result.missing_keys))
    print("  unexpected_keys=", len(result.unexpected_keys))

    if strict:
        print("[WARN] 你启用了 strict=True；只有完全匹配时才算真正通过。")

    if len(filtered_sd) == 0:
        print("[FAIL] 没有任何参数能够成功对齐加载。")
    elif len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
        print("[OK] checkpoint 与当前模型完全匹配。")
    else:
        print("[WARN] checkpoint 只做到了部分匹配加载，说明模型定义或配置可能与训练时不完全一致。")

    if missing_after_filter:
        print("\n过滤后仍未覆盖的前 20 个模型键:")
        for k in missing_after_filter[:20]:
            print("  -", k)

    if unexpected_after_filter:
        print("\n过滤后 unexpected 键（理论上一般为空）:")
        for k in unexpected_after_filter[:20]:
            print("  -", k)


def run_forward_smoke_test(model, device, batch_size: int = 1, image_size: int = 256) -> None:
    import torch
    import sde_lib
    from models import utils as mutils
    from configs.vp import AI4Scup2_ddpm_continuous as configs

    print_header("[4] 前向接口检查（不做采样）")

    # 用 notebook 里的典型形状：[B, 1, 256, 256]
    x = torch.randn(batch_size, 1, image_size, image_size, device=device)
    labels = torch.ones(batch_size, device=device) * 999.0

    with torch.no_grad():
        out = model(x, labels)

    print(f"输入 x shape     : {tuple(x.shape)}")
    print(f"输入 labels shape: {tuple(labels.shape)}")
    print(f"输出 out shape   : {tuple(out.shape)}")
    print(f"输出 dtype       : {out.dtype}")
    print(f"输出 device      : {out.device}")
    print(f"输出均值/方差    : mean={out.mean().item():.6f}, std={out.std().item():.6f}")

    # 再检查 score_fn 这一层是否也能通
    config = configs.get_config()
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=config.training.continuous)

    t = torch.ones(batch_size, device=device) * 0.1
    with torch.no_grad():
        score = score_fn(x, t)

    print(f"score(x,t) shape : {tuple(score.shape)}")
    print(f"score 均值/方差  : mean={score.mean().item():.6f}, std={score.std().item():.6f}")
    print("[OK] 前向接口检查通过：模型 forward 和 score_fn 都能跑通。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, required=True, help="USCT_download 根目录")
    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint 文件路径，可为空")
    parser.add_argument("--device", type=str, default="auto", help="例如 cpu / cuda:0 / auto")
    parser.add_argument("--no_dataparallel", action="store_true", help="禁用 DataParallel")
    parser.add_argument("--strict", action="store_true", help="按 strict=True 方式加载")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    print_header("[0] 基本信息")
    print("repo_root  =", args.repo_root)
    print("checkpoint =", args.checkpoint if args.checkpoint else "<未提供>")
    print("device arg =", args.device)

    check_basic_dependencies()

    import torch

    repo_root = Path(args.repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root 不存在: {repo_root}")

    ensure_repo_on_path(repo_root)
    device = resolve_device(args.device)

    print(f"torch version = {torch.__version__}")
    print(f"cuda available= {torch.cuda.is_available()}")
    print(f"resolved device= {device}")

    print_header("[1] 导入配置")
    config = build_config(device=device, use_dataparallel=not args.no_dataparallel)
    print("config.model.name        =", config.model.name)
    print("config.data.image_size   =", config.data.image_size)
    print("config.data.num_channels =", config.data.num_channels)
    print("config.model.nf          =", config.model.nf)
    print("config.model.ch_mult     =", tuple(config.model.ch_mult))
    print("config.model.num_scales  =", config.model.num_scales)
    print("config.training.continuous =", config.training.continuous)
    print("config.device_ids        =", list(config.device_ids))

    print_header("[2] 构建 score model")
    model, base_model = build_model(config, device=device, use_dataparallel=not args.no_dataparallel)
    num_params = sum(p.numel() for p in base_model.parameters())
    print("model class   =", model.__class__.__name__)
    print("base model    =", base_model.__class__.__name__)
    print("num params    =", num_params)
    print("first key     =", next(iter(model.state_dict().keys())))

    if args.checkpoint:
        load_checkpoint(model, Path(args.checkpoint), device=device, strict=args.strict)
    else:
        print_header("[3] 跳过 checkpoint")
        print("未提供 checkpoint，已跳过加载检查。")

    run_forward_smoke_test(model, device=device, batch_size=args.batch_size, image_size=args.image_size)

    print_header("[完成]")
    print("脚本执行结束。你现在可以明确判断：")
    print("1) 配置能否导入")
    print("2) 模型能否构建")
    print("3) checkpoint 是否存在")
    print("4) checkpoint 与模型是否匹配")
    print("5) forward / score_fn 接口是否跑通")


if __name__ == "__main__":
    main()
