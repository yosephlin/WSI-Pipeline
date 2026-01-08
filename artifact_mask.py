from __future__ import annotations

import math
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np

__all__ = [
    "GRANDQC_CLASS_NAMES",
    "GRANDQC_COLORS_QC7",
    "GRANDQC_CLEAN_TISSUE_CLASS",
    "GRANDQC_BACKGROUND_CLASS",
    "GRANDQC_ARTIFACT_CLASS_IDS",
    "load_grandqc_artifact_model",
    "resize_bool_mask",
    "resize_label_mask",
    "downsample_mask_nearest_by_shape",
    "colorize_grandqc_qc7",
    "colorize_grandqc_artifacts",
    "make_grandqc_artifact_legend_image",
    "segment_grandqc_artifacts_rgb",
    "segment_grandqc_artifacts_slide",
    "segment_grandqc_artifacts_thumbnail",
]

GRANDQC_CLEAN_TISSUE_CLASS = 1
GRANDQC_BACKGROUND_CLASS = 7
GRANDQC_ARTIFACT_CLASS_IDS = (2, 3, 4, 5, 6)

# Class labels per GrandQC README:
#   1: Normal Tissue
#   2: Fold
#   3: Darkspot & Foreign Object
#   4: PenMarking
#   5: Edge & Air Bubble
#   6: OOF (Out of Focus)
#   7: Background
GRANDQC_CLASS_NAMES: Dict[int, str] = {
    1: "Clean tissue",
    2: "Tissue fold",
    3: "Dark spot / foreign object",
    4: "Pen marking",
    5: "Edge / air bubble",
    6: "Out of focus",
    7: "Background",
}

# Colors per grandqc/01_WSI_inference_OPENSLIDE_QC/wsi_colors.py (RGB order).
# Index i corresponds to class (i+1).
GRANDQC_COLORS_QC7: Tuple[Tuple[int, int, int], ...] = (
    (128, 128, 128),  # 1: clean tissue
    (255, 99, 71),  # 2: fold
    (0, 255, 0),  # 3: dark spot / foreign object
    (255, 0, 0),  # 4: pen
    (255, 0, 255),  # 5: edge / air bubble
    (75, 0, 130),  # 6: out of focus
    (255, 255, 255),  # 7: background
)


def _estimate_level0_mpp(slide: Any) -> Optional[float]:
    props = getattr(slide, "properties", None) or {}
    candidates = []
    for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
        try:
            v = props.get(key)
        except Exception:
            v = None
        if v is None:
            continue
        try:
            candidates.append(float(v))
        except Exception:
            continue
    if not candidates:
        return None
    return float(sum(candidates) / len(candidates))


def _install_timm_legacy_aliases() -> None:
    """
    Older GrandQC checkpoints may reference legacy timm module paths.
    """
    import importlib
    import importlib.abc
    import importlib.util
    import sys

    class _AliasLoader(importlib.abc.Loader):
        def __init__(self, target_name: str):
            self.target_name = target_name

        def create_module(self, spec):  # type: ignore[override]
            return None

        def exec_module(self, module):  # type: ignore[override]
            target = importlib.import_module(self.target_name)
            sys.modules[module.__name__] = target

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # type: ignore[override]
            def _safe_find_spec(name: str):
                try:
                    return importlib.util.find_spec(name)
                except ModuleNotFoundError:
                    return None

            if fullname.startswith("timm.models.layers."):
                suffix = fullname.split("timm.models.layers.", 1)[1]
                target_name = f"timm.layers.{suffix}"
                if _safe_find_spec(target_name) is not None:
                    return importlib.util.spec_from_loader(fullname, _AliasLoader(target_name))
            if fullname == "timm.models.efficientnet_blocks":
                target_name = "timm.models._efficientnet_blocks"
                if _safe_find_spec(target_name) is not None:
                    return importlib.util.spec_from_loader(fullname, _AliasLoader(target_name))
            return None

    if not any(type(f).__name__ == "_Finder" for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())

    # If timm.layers is missing (older timm), alias to timm.models.layers for SMP.
    try:
        has_timm_layers = importlib.util.find_spec("timm.layers") is not None
    except ModuleNotFoundError:
        has_timm_layers = False
    try:
        has_timm_models_layers = importlib.util.find_spec("timm.models.layers") is not None
    except ModuleNotFoundError:
        has_timm_models_layers = False
    if (not has_timm_layers) and has_timm_models_layers:
        try:
            ml = importlib.import_module("timm.models.layers")
            sys.modules.setdefault("timm.layers", ml)
            try:
                act = importlib.import_module("timm.models.layers.activations")
                sys.modules.setdefault("timm.layers.activations", act)
            except Exception:
                pass
            try:
                helpers = importlib.import_module("timm.models.layers.helpers")
                sys.modules.setdefault("timm.layers.helpers", helpers)
            except Exception:
                pass
        except Exception:
            pass


def _install_smp_legacy_aliases() -> None:
    """
    Some GrandQC checkpoints were saved with older `segmentation_models_pytorch` symbols.

    Newer SMP versions (e.g. 0.5.x) renamed Unet blocks:
      - DecoderBlock -> UnetDecoderBlock
      - CenterBlock  -> UnetCenterBlock
    """
    try:
        import segmentation_models_pytorch.decoders.unet.decoder as unet_decoder
    except Exception:
        return

    if (not hasattr(unet_decoder, "DecoderBlock")) and hasattr(unet_decoder, "UnetDecoderBlock"):
        setattr(unet_decoder, "DecoderBlock", getattr(unet_decoder, "UnetDecoderBlock"))
    if (not hasattr(unet_decoder, "CenterBlock")) and hasattr(unet_decoder, "UnetCenterBlock"):
        setattr(unet_decoder, "CenterBlock", getattr(unet_decoder, "UnetCenterBlock"))


def _patch_timm_checkpoint_modules(model: Any) -> None:
    """
    Torch pickled checkpoints bypass `__init__` on load, so modules may be missing new attributes
    referenced by newer library versions. Patch the common ones we rely on.
    """
    try:
        import timm.models._efficientnet_blocks as eff
    except Exception:
        return

    try:
        import torch.nn as nn
    except Exception:
        return

    depthwise = getattr(eff, "DepthwiseSeparableConv", None)
    inverted = getattr(eff, "InvertedResidual", None)
    conv_bn_act = getattr(eff, "ConvBnAct", None)
    edge = getattr(eff, "EdgeResidual", None)

    block_types = tuple(t for t in (depthwise, inverted, conv_bn_act, edge) if t is not None)
    if not block_types:
        return

    try:
        from timm.layers import DropPath
    except Exception:  # pragma: no cover
        try:
            from timm.models.layers import DropPath  # type: ignore
        except Exception:
            DropPath = None  # type: ignore[assignment]

    try:
        modules = model.modules()
    except Exception:
        return
    for m in modules:
        try:
            if not isinstance(m, block_types):
                continue
        except Exception:
            continue
        if not hasattr(m, "conv_s2d"):
            setattr(m, "conv_s2d", None)
        if not hasattr(m, "bn_s2d"):
            setattr(m, "bn_s2d", None)
        if not hasattr(m, "aa"):
            setattr(m, "aa", nn.Identity())
        if not hasattr(m, "has_skip"):
            if hasattr(m, "has_residual"):
                setattr(m, "has_skip", bool(getattr(m, "has_residual")))
            else:
                setattr(m, "has_skip", False)
        if not hasattr(m, "drop_path"):
            dp_rate = float(getattr(m, "drop_path_rate", 0.0) or 0.0)
            if (DropPath is not None) and dp_rate > 0:
                setattr(m, "drop_path", DropPath(dp_rate))
            else:
                setattr(m, "drop_path", nn.Identity())


def load_grandqc_artifact_model(
    model_path: str | Path,
    *,
    device: str = "cpu",
    encoder_name: str = "timm-efficientnet-b0",
) -> Tuple[Any, Any]:
    """
    Load GrandQC artifact segmentation model and preprocessing function.

    Returns: (model, preprocessing_fn)
    """
    _install_timm_legacy_aliases()
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "GrandQC artifact segmentation requires PyTorch (`torch`). Install it."
        ) from e

    try:
        import segmentation_models_pytorch as smp
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "GrandQC artifact segmentation requires `segmentation-models-pytorch` "
            "(import name `segmentation_models_pytorch`). Install it."
        ) from e

    _install_smp_legacy_aliases()

    model_path = str(model_path)
    device_t = torch.device(str(device))

    def _ensure_wsi_feature_field_on_path() -> None:
        try:
            import wsi_feature_field  # noqa: F401
            return
        except ModuleNotFoundError:
            pass
        repo_root = Path(__file__).resolve().parents[1]
        src_path = repo_root / "src"
        if src_path.exists():
            src_str = str(src_path)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)

    def _extract_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                val = obj.get(key)
                if isinstance(val, dict):
                    return val
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj  # raw state_dict
        if hasattr(obj, "state_dict"):
            try:
                state = obj.state_dict()  # type: ignore[union-attr]
                if isinstance(state, dict):
                    return state
            except Exception:
                return None
        return None

    checkpoint: Any = None
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        _ensure_wsi_feature_field_on_path()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except pickle.UnpicklingError:
        checkpoint = None

    if checkpoint is None:
        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None  # type: ignore[assignment]

        if safe_globals is not None:
            try:
                with safe_globals([smp.Unet]):
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception:
                checkpoint = None
        else:
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception:
                checkpoint = None

    if checkpoint is None:
        raise ImportError(
            "Failed to load GrandQC checkpoint; could not resolve wsi_feature_field or "
            "weights-only loading."
        )

    state_dict = _extract_state_dict(checkpoint)
    if state_dict is not None:
        classes = int(state_dict["segmentation_head.0.weight"].shape[0])
        rebuilt = smp.Unet(
            encoder_name=str(encoder_name),
            encoder_weights=None,
            in_channels=3,
            classes=classes,
            activation=None,
        )
        rebuilt.load_state_dict(state_dict, strict=True)
        model = rebuilt
    else:
        model = checkpoint
        _patch_timm_checkpoint_modules(model)

    try:
        model.to(device_t)
    except Exception:
        pass
    try:
        model.eval()
    except Exception:
        pass

    preprocessing_fn = smp.encoders.get_preprocessing_fn(str(encoder_name), "imagenet")
    return model, preprocessing_fn


def resize_bool_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """
    Nearest-neighbor resize for boolean masks.
    """
    h, w = (int(v) for v in size_hw)
    if h <= 0 or w <= 0:
        raise ValueError(f"size_hw must be positive, got {size_hw}")
    mask = np.asarray(mask).astype(bool, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    try:
        import cv2

        m = (mask.astype(np.uint8) * 255).astype(np.uint8, copy=False)
        out = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return (out > 0).astype(bool)
    except Exception:
        from PIL import Image

        m = (mask.astype(np.uint8) * 255).astype(np.uint8, copy=False)
        pil = Image.fromarray(m)
        pil = pil.resize((w, h), resample=Image.Resampling.NEAREST)
        out = np.asarray(pil, dtype=np.uint8)
    return (out > 0).astype(bool)


def resize_label_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """
    Nearest-neighbor resize for integer label masks.
    """
    h, w = (int(v) for v in size_hw)
    if h <= 0 or w <= 0:
        raise ValueError(f"size_hw must be positive, got {size_hw}")
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got {m.shape}")
    if m.shape[0] == h and m.shape[1] == w:
        return m
    try:
        import cv2

        out = cv2.resize(m.astype(np.int32, copy=False), (w, h), interpolation=cv2.INTER_NEAREST)
        return out.astype(m.dtype, copy=False)
    except Exception:
        from PIL import Image

        pil = Image.fromarray(m.astype(np.int32, copy=False))
        pil = pil.resize((w, h), resample=Image.Resampling.NEAREST)
        out = np.asarray(pil, dtype=np.int32)
        return out.astype(m.dtype, copy=False)


def downsample_mask_nearest_by_shape(mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Fast nearest-neighbor downsample/upsample based on output shape using center-of-pixel mapping.

    This is deterministic and avoids OpenCV/Pillow dependency for simple resampling.
    """
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid out_hw={out_hw}")

    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got {m.shape}")

    in_h, in_w = int(m.shape[0]), int(m.shape[1])
    if (in_h, in_w) == (out_h, out_w):
        return m

    scale_y = in_h / float(out_h)
    scale_x = in_w / float(out_w)
    ys = np.clip(((np.arange(out_h) + 0.5) * scale_y - 0.5).astype(np.int64), 0, in_h - 1)
    xs = np.clip(((np.arange(out_w) + 0.5) * scale_x - 0.5).astype(np.int64), 0, in_w - 1)
    return m[np.ix_(ys, xs)]


def colorize_grandqc_qc7(
    class_map: np.ndarray,
    *,
    show_clean_tissue: bool = True,
    show_background: bool = True,
    background_rgb: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Convert a GrandQC class map (1..7) to an RGB visualization using the official QC7 colors.
    """
    cm = np.asarray(class_map)
    if cm.ndim != 2:
        raise ValueError(f"class_map must be 2D, got {cm.shape}")

    h, w = (int(cm.shape[0]), int(cm.shape[1]))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = int(background_rgb[0])
    out[:, :, 1] = int(background_rgb[1])
    out[:, :, 2] = int(background_rgb[2])

    for class_id, rgb in enumerate(GRANDQC_COLORS_QC7, start=1):
        if class_id == GRANDQC_CLEAN_TISSUE_CLASS and not bool(show_clean_tissue):
            continue
        if class_id == GRANDQC_BACKGROUND_CLASS and not bool(show_background):
            continue
        m = cm == int(class_id)
        if not np.any(m):
            continue
        out[m, 0] = int(rgb[0])
        out[m, 1] = int(rgb[1])
        out[m, 2] = int(rgb[2])

    return out


def colorize_grandqc_artifacts(
    class_map: np.ndarray,
    *,
    background_rgb: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Artifact-only visualization: colorize only GrandQC artifact classes (2..6).

    Clean tissue (1) and background (7) are set to `background_rgb`.
    """
    cm = np.asarray(class_map)
    if cm.ndim != 2:
        raise ValueError(f"class_map must be 2D, got {cm.shape}")

    h, w = (int(cm.shape[0]), int(cm.shape[1]))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = int(background_rgb[0])
    out[:, :, 1] = int(background_rgb[1])
    out[:, :, 2] = int(background_rgb[2])

    for class_id in GRANDQC_ARTIFACT_CLASS_IDS:
        rgb = GRANDQC_COLORS_QC7[int(class_id) - 1]
        m = cm == int(class_id)
        if not np.any(m):
            continue
        out[m, 0] = int(rgb[0])
        out[m, 1] = int(rgb[1])
        out[m, 2] = int(rgb[2])

    return out


def make_grandqc_artifact_legend_image(
    *,
    class_ids: Sequence[int] = GRANDQC_ARTIFACT_CLASS_IDS,
    include_hidden_entry: bool = True,
    hidden_label: str = "Clean tissue / background (hidden)",
    hidden_rgb: Tuple[int, int, int] = (0, 0, 0),
    swatch_size: int = 18,
    gap: int = 6,
    margin: int = 8,
    font_size: int = 13,
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
    text_rgb: Tuple[int, int, int] = (0, 0, 0),
    outline_rgb: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Render a compact legend for GrandQC artifact classes (2..6 by default).

    Returns an RGB uint8 image suitable for Streamlit display or saving via Pillow.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow is required to render legend images.") from e

    items: List[Tuple[Tuple[int, int, int], str]] = []
    if include_hidden_entry:
        items.append((tuple(int(v) for v in hidden_rgb), str(hidden_label)))

    for cid in class_ids:
        cid_i = int(cid)
        if cid_i < 1 or cid_i > len(GRANDQC_COLORS_QC7):
            continue
        rgb = GRANDQC_COLORS_QC7[cid_i - 1]
        name = GRANDQC_CLASS_NAMES.get(cid_i, f"class {cid_i}")
        items.append((rgb, f"{cid_i}: {name}"))

    if not items:
        out = np.zeros((1, 1, 3), dtype=np.uint8)
        out[0, 0] = np.asarray(bg_rgb, dtype=np.uint8)
        return out

    swatch_size = max(4, int(swatch_size))
    gap = max(0, int(gap))
    margin = max(0, int(margin))
    font_size = max(6, int(font_size))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:  # pragma: no cover
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    def _text_size(text: str) -> Tuple[int, int]:
        if hasattr(draw, "textbbox"):
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return int(r - l), int(b - t)
        # Pillow<8 fallback
        w, h = draw.textsize(text, font=font)  # type: ignore[attr-defined]
        return int(w), int(h)

    text_sizes = [_text_size(label) for _, label in items]
    max_text_w = max(w for w, _ in text_sizes)
    max_text_h = max(h for _, h in text_sizes)
    row_h = max(swatch_size, max_text_h)

    width = margin * 2 + swatch_size + gap + max_text_w
    height = margin * 2 + len(items) * row_h + (len(items) - 1) * gap

    img = Image.new("RGB", (int(width), int(height)), color=tuple(int(v) for v in bg_rgb))
    draw = ImageDraw.Draw(img)

    x0 = margin
    x_text = margin + swatch_size + gap
    y = margin
    for (rgb, label), (_, th) in zip(items, text_sizes):
        draw.rectangle(
            [x0, y, x0 + swatch_size - 1, y + swatch_size - 1],
            fill=tuple(int(v) for v in rgb),
            outline=tuple(int(v) for v in outline_rgb),
        )
        ty = y + int((row_h - th) // 2)
        draw.text((x_text, ty), label, fill=tuple(int(v) for v in text_rgb), font=font)
        y += row_h + gap

    return np.asarray(img, dtype=np.uint8)


def _to_tensor_x(image: np.ndarray) -> np.ndarray:
    return image.transpose(2, 0, 1).astype("float32", copy=False)


def segment_grandqc_artifacts_rgb(
    rgb: np.ndarray,
    *,
    tissue_mask: Optional[np.ndarray],
    model: Any,
    preprocessing_fn: Any,
    device: str = "cpu",
    input_size: int = 512,
    back_class: int = 7,
    artifact_class_ids: Sequence[int] = (2, 3, 4, 5, 6),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Run GrandQC artifact segmentation on an RGB image.

    Inputs:
      - rgb: uint8 [H,W,3] at any resolution; resized to `input_size`.
      - tissue_mask: optional bool mask (same size as `rgb` OR same size as `input_size`)
                    used to gate predictions to tissue only.

    Returns:
      (artifact_mask_bool[input_size,input_size], class_map_int16[input_size,input_size], meta)
    """
    import torch

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected rgb [H,W,3], got shape {rgb.shape}")
    input_size = int(input_size)
    if input_size <= 0:
        raise ValueError(f"input_size must be > 0, got {input_size}")

    img_u8 = rgb.astype(np.uint8, copy=False)
    if img_u8.shape[0] != input_size or img_u8.shape[1] != input_size:
        try:
            import cv2

            interp = cv2.INTER_AREA if (img_u8.shape[0] > input_size or img_u8.shape[1] > input_size) else cv2.INTER_LINEAR
            img_u8 = cv2.resize(img_u8, (input_size, input_size), interpolation=interp).astype(np.uint8, copy=False)
        except Exception:
            from PIL import Image

            pil = Image.fromarray(img_u8)
            pil = pil.resize((input_size, input_size), Image.Resampling.LANCZOS)
            img_u8 = np.asarray(pil, dtype=np.uint8)

    tm = None
    if tissue_mask is not None:
        tm_arr = np.asarray(tissue_mask).astype(bool, copy=False)
        if tm_arr.ndim != 2:
            raise ValueError(f"tissue_mask must be 2D, got {tm_arr.shape}")
        if tm_arr.shape[0] == input_size and tm_arr.shape[1] == input_size:
            tm = tm_arr
        else:
            # Assume it's in the original rgb resolution; resize to input_size.
            tm = resize_bool_mask(tm_arr, (input_size, input_size))

    x = preprocessing_fn(img_u8)
    x = _to_tensor_x(x)
    x_tensor = torch.from_numpy(x).to(device).unsqueeze(0)

    with torch.inference_mode():
        pred = model.predict(x_tensor) if hasattr(model, "predict") else model(x_tensor)
    pred_np = pred.detach().to("cpu").numpy()
    if pred_np.ndim == 4:
        pred_np = pred_np[0]  # [C,H,W]
    if pred_np.ndim != 3:
        raise RuntimeError(f"Unexpected GrandQC output shape: {pred_np.shape}")

    class_map = pred_np.argmax(axis=0).astype(np.int16)  # [H,W]
    # Heuristic: some checkpoints may produce 0..6; align to 1..7 class ids.
    if int(class_map.min()) == 0 and int(class_map.max()) == 6:
        class_map = class_map + 1

    if tm is not None:
        if tm.shape != class_map.shape:
            raise ValueError(f"tissue_mask shape {tm.shape} != class_map shape {class_map.shape}")
        class_map = class_map.copy()
        class_map[~tm] = int(back_class)

    artifact_mask = np.isin(class_map, list(artifact_class_ids)).astype(bool)

    meta: Dict[str, Any] = {
        "grandqc_input_size": int(input_size),
        "grandqc_back_class": int(back_class),
        "grandqc_artifact_class_ids": [int(v) for v in artifact_class_ids],
    }
    return artifact_mask, class_map, meta


def segment_grandqc_artifacts_thumbnail(
    thumb_rgb: np.ndarray,
    thumb_tissue_mask: np.ndarray,
    *,
    model: Any,
    preprocessing_fn: Any,
    device: str = "cpu",
    input_size: int = 512,
    back_class: int = 7,
    artifact_class_ids: Sequence[int] = (2, 3, 4, 5, 6),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience wrapper: segment artifacts on a thumbnail and return a mask at thumbnail resolution.

    Returns:
      (artifact_mask_thumb[H,W], class_map_512[512,512], meta)
    """
    artifact_512, class_map_512, meta = segment_grandqc_artifacts_rgb(
        thumb_rgb,
        tissue_mask=thumb_tissue_mask,
        model=model,
        preprocessing_fn=preprocessing_fn,
        device=device,
        input_size=int(input_size),
        back_class=int(back_class),
        artifact_class_ids=artifact_class_ids,
    )
    artifact_thumb = resize_bool_mask(artifact_512, (int(thumb_rgb.shape[0]), int(thumb_rgb.shape[1])))
    return artifact_thumb, class_map_512, meta


def segment_grandqc_artifacts_slide(
    slide: Any,
    thumb_tissue_mask: np.ndarray,
    *,
    model: Any,
    preprocessing_fn: Any,
    device: str = "cpu",
    artifact_mpp_model: float = 1.5,
    patch_size: int = 512,
    min_tissue_pixels: int = 50,
    batch_size: int = 4,
    back_class: int = 7,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Patch-based GrandQC artifact segmentation (matches upstream usage).

    This runs inference on the WSI at the model's expected physical resolution (MPP),
    stitches a label map in "QC-space" (MPP=artifact_mpp_model), then downsamples
    to the thumbnail/tissue-mask resolution for visualization + downstream masking.

    Returns:
      (artifact_mask_thumb[H,W], class_map_thumb[H,W], meta)
    """
    import torch

    tm_thumb = np.asarray(thumb_tissue_mask).astype(bool, copy=False)
    if tm_thumb.ndim != 2:
        raise ValueError(f"thumb_tissue_mask must be 2D, got {tm_thumb.shape}")

    level0_mpp = _estimate_level0_mpp(slide)
    if level0_mpp is None or level0_mpp <= 0:
        raise ValueError("OpenSlide missing MPP metadata (openslide.mpp-x/openslide.mpp-y/aperio.MPP).")

    w0, h0 = (int(v) for v in slide.level_dimensions[0])  # (W,H)
    qc_w = max(1, int(round(float(w0) * float(level0_mpp) / float(artifact_mpp_model))))
    qc_h = max(1, int(round(float(h0) * float(level0_mpp) / float(artifact_mpp_model))))

    tm_qc = resize_bool_mask(tm_thumb, (qc_h, qc_w))
    tissue_raw = np.where(tm_qc, 0, 1).astype(np.uint8, copy=False)  # 0=tissue, 1=background

    ys, xs = np.where(tm_qc)
    thumb_h, thumb_w = (int(tm_thumb.shape[0]), int(tm_thumb.shape[1]))
    if ys.size == 0 or xs.size == 0:
        class_map_thumb = np.full((thumb_h, thumb_w), int(back_class), dtype=np.uint8)
        artifact_mask_thumb = np.zeros((thumb_h, thumb_w), dtype=bool)
        meta: Dict[str, Any] = {
            "artifact_mpp_model": float(artifact_mpp_model),
            "qc_shape": [int(qc_h), int(qc_w)],
            "note": "No tissue detected; artifact mask empty.",
        }
        return artifact_mask_thumb, class_map_thumb, meta

    # Tissue bbox in QC-space (restrict work + memory).
    y0_roi = int(ys.min())
    y1_roi = int(ys.max()) + 1
    x0_roi = int(xs.min())
    x1_roi = int(xs.max()) + 1
    roi_h = int(y1_roi - y0_roi)
    roi_w = int(x1_roi - x0_roi)

    qc_roi = np.full((roi_h, roi_w), int(back_class), dtype=np.uint8)

    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    he_start = max(0, y0_roi // patch_size)
    he_end = min(int(math.ceil(qc_h / patch_size)), int(math.ceil(y1_roi / patch_size)))
    wi_start = max(0, x0_roi // patch_size)
    wi_end = min(int(math.ceil(qc_w / patch_size)), int(math.ceil(x1_roi / patch_size)))

    level0_per_qc_px = float(artifact_mpp_model) / float(level0_mpp)
    read_size_l0 = max(1, int(round(float(patch_size) * level0_per_qc_px)))

    read_level = 0
    read_size_level = int(read_size_l0)
    try:
        read_level = int(slide.get_best_level_for_downsample(level0_per_qc_px))
        level_down = float(slide.level_downsamples[int(read_level)])
        if level_down > 0:
            read_size_level = max(1, int(round(float(read_size_l0) / float(level_down))))
    except Exception:
        read_level = 0
        read_size_level = int(read_size_l0)

    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow is required for GrandQC patch resizing.") from e

    device_t = torch.device(str(device))

    batch_imgs: List[np.ndarray] = []
    batch_infos: List[Tuple[int, int, int, int, int, int, np.ndarray]] = []
    # (tile_y0, tile_y1, tile_x0, tile_x1, int_y0, int_x0, td_patch_padded)

    num_tiles_total = 0
    num_tiles_infer = 0
    num_tiles_skip = 0
    did_shift_labels = False

    def _flush() -> None:
        nonlocal num_tiles_infer, did_shift_labels
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).to(device_t)

        use_autocast = str(device).startswith("cuda")
        autocast_ctx = nullcontext()
        if use_autocast and hasattr(torch, "autocast"):
            try:
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            except TypeError:  # pragma: no cover
                autocast_ctx = torch.autocast("cuda", dtype=torch.float16)

        with torch.inference_mode(), autocast_ctx:
            pred = model.predict(x) if hasattr(model, "predict") else model(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if not isinstance(pred, torch.Tensor):
            raise RuntimeError(f"Unexpected model output type: {type(pred)}")
        if pred.ndim == 3:  # [C,H,W]
            pred = pred.unsqueeze(0)
        if pred.ndim != 4:
            raise RuntimeError(f"Unexpected model output shape: {tuple(pred.shape)}")

        labels = pred.argmax(dim=1).to("cpu").numpy().astype(np.int16, copy=False)  # [B,H,W]

        for (tile_y0, tile_y1, tile_x0, tile_x1, int_y0, int_x0, td_pad), lbl in zip(batch_infos, labels, strict=True):
            # If checkpoint emits 0..6, align to 1..7 (GrandQC README class ids).
            if int(lbl.min()) == 0 and int(lbl.max()) == 6:
                lbl = lbl + 1
                did_shift_labels = True

            lbl = np.asarray(lbl, dtype=np.int16, copy=False)
            # Force background where tissue-mask says background.
            lbl = np.where(td_pad == 1, int(back_class), lbl)
            # Safety: treat label 0 as background if it appears.
            lbl = np.where(lbl == 0, int(back_class), lbl)

            tile_h = int(tile_y1 - tile_y0)
            tile_w = int(tile_x1 - tile_x0)
            lbl = lbl[:tile_h, :tile_w]

            # Intersection of tile with ROI bbox.
            y0_int = max(int(tile_y0), int(y0_roi))
            y1_int = min(int(tile_y1), int(y1_roi))
            x0_int = max(int(tile_x0), int(x0_roi))
            x1_int = min(int(tile_x1), int(x1_roi))
            if y1_int <= y0_int or x1_int <= x0_int:
                continue

            ry0 = y0_int - int(y0_roi)
            ry1 = y1_int - int(y0_roi)
            rx0 = x0_int - int(x0_roi)
            rx1 = x1_int - int(x0_roi)

            ly0 = y0_int - int(tile_y0)
            ly1 = y1_int - int(tile_y0)
            lx0 = x0_int - int(tile_x0)
            lx1 = x1_int - int(tile_x0)

            qc_roi[ry0:ry1, rx0:rx1] = lbl[ly0:ly1, lx0:lx1].astype(np.uint8, copy=False)
            num_tiles_infer += 1

        batch_imgs.clear()
        batch_infos.clear()

    for he in range(int(he_start), int(he_end)):
        for wi in range(int(wi_start), int(wi_end)):
            num_tiles_total += 1
            tile_x0 = int(wi * patch_size)
            tile_y0 = int(he * patch_size)
            tile_x1 = min(tile_x0 + patch_size, int(qc_w))
            tile_y1 = min(tile_y0 + patch_size, int(qc_h))

            td_patch = tissue_raw[tile_y0:tile_y1, tile_x0:tile_x1]
            if int(np.count_nonzero(td_patch == 0)) <= int(min_tissue_pixels):
                num_tiles_skip += 1
                continue

            td_pad = np.ones((patch_size, patch_size), dtype=np.uint8)
            td_pad[: int(tile_y1 - tile_y0), : int(tile_x1 - tile_x0)] = td_patch

            x0_l0 = int(round(float(tile_x0) * level0_per_qc_px))
            y0_l0 = int(round(float(tile_y0) * level0_per_qc_px))

            patch = slide.read_region((x0_l0, y0_l0), int(read_level), (int(read_size_level), int(read_size_level))).convert("RGB")
            patch = patch.resize((patch_size, patch_size), resample=Image.Resampling.LANCZOS)

            img_u8 = np.asarray(patch, dtype=np.uint8)
            x_pre = preprocessing_fn(img_u8)
            x_pre = _to_tensor_x(x_pre)

            batch_imgs.append(x_pre)
            batch_infos.append((tile_y0, tile_y1, tile_x0, tile_x1, 0, 0, td_pad))

            if len(batch_imgs) >= int(batch_size):
                _flush()

    _flush()

    # Downsample QC-space ROI labels to thumbnail resolution (fast nearest via index mapping).
    ys_thumb = np.clip(
        ((np.arange(thumb_h) + 0.5) * (qc_h / float(thumb_h)) - 0.5).astype(np.int64),
        0,
        qc_h - 1,
    )
    xs_thumb = np.clip(
        ((np.arange(thumb_w) + 0.5) * (qc_w / float(thumb_w)) - 0.5).astype(np.int64),
        0,
        qc_w - 1,
    )
    ys_in = (ys_thumb >= int(y0_roi)) & (ys_thumb < int(y1_roi))
    xs_in = (xs_thumb >= int(x0_roi)) & (xs_thumb < int(x1_roi))

    class_map_thumb = np.full((thumb_h, thumb_w), int(back_class), dtype=np.uint8)
    if bool(np.any(ys_in)) and bool(np.any(xs_in)):
        out_y = np.where(ys_in)[0]
        out_x = np.where(xs_in)[0]
        roi_y = (ys_thumb[ys_in] - int(y0_roi)).astype(np.int64, copy=False)
        roi_x = (xs_thumb[xs_in] - int(x0_roi)).astype(np.int64, copy=False)
        class_map_thumb[np.ix_(out_y, out_x)] = qc_roi[np.ix_(roi_y, roi_x)]

    artifact_mask_thumb = np.isin(class_map_thumb, np.asarray(GRANDQC_ARTIFACT_CLASS_IDS, dtype=np.uint8))

    meta = {
        "artifact_mpp_model": float(artifact_mpp_model),
        "qc_shape": [int(qc_h), int(qc_w)],
        "qc_roi_bbox_xyxy": [int(x0_roi), int(y0_roi), int(x1_roi), int(y1_roi)],
        "patch_size": int(patch_size),
        "min_tissue_pixels": int(min_tissue_pixels),
        "batch_size": int(batch_size),
        "read_level": int(read_level),
        "read_size_level": int(read_size_level),
        "level0_mpp": float(level0_mpp),
        "level0_per_qc_px": float(level0_per_qc_px),
        "tiles_total": int(num_tiles_total),
        "tiles_skipped": int(num_tiles_skip),
        "tiles_inferred": int(num_tiles_infer),
        "label_shifted_0to6_to_1to7": bool(did_shift_labels),
    }
    return artifact_mask_thumb.astype(bool, copy=False), class_map_thumb, meta
