from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from roi_sampler import ROICandidate, pick_roi_pair, sample_candidate_rois_selective
from slide_vit import SlideViT, SlideViTConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ViewConfig:
    roi_size: int = 16
    global_crop_size: int = 14
    local_crop_size: int = 6
    num_global: int = 2
    num_local: int = 10
    flip_prob: float = 0.5
    posterize_step: float = 0.05
    posterize_noise_std: float = 0.005
    posterize_prob: float = 0.5
    student_mask_ratio_global: float = 0.45
    student_mask_ratio_local: float = 0.55
    teacher_mask_ratio_global: float = 0.0
    teacher_mask_ratio_local: float = 0.0


@dataclass
class CropView:
    roi_id: str
    view_type: str
    features: np.ndarray
    valid_mask: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int]


class SlideViTiBOT(nn.Module):
    def __init__(self, config: SlideViTConfig, num_prototypes: int, head_hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.student = SlideViT(config)
        self.teacher = SlideViT(config)
        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.student_head = IBotHead(config.embed_dim, num_prototypes, head_hidden_dim)
        self.teacher_head = IBotHead(config.embed_dim, num_prototypes, head_hidden_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        self.mask_token = nn.Parameter(torch.zeros(config.input_dim))
        self.register_buffer("center_token", torch.zeros(num_prototypes))
        self.register_buffer("center_cls", torch.zeros(num_prototypes))

    def forward_student(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask.any():
            x = x.clone()
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = torch.where(mask_expanded, self.mask_token.view(1, 1, 1, -1), x)
        tokens, pooled = self.student.forward_features(x, valid_mask=valid_mask)
        token_logits = self.student_head(tokens)
        cls_logits = self.student_head(pooled)
        return token_logits, cls_logits

    def forward_teacher(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens, pooled = self.teacher.forward_features(x, valid_mask=valid_mask)
            token_logits = self.teacher_head(tokens)
            cls_logits = self.teacher_head(pooled)
        return token_logits, cls_logits

    def ema_update(self, momentum: float) -> None:
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
            for t_param, s_param in zip(self.teacher_head.parameters(), self.student_head.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


class IBotHead(nn.Module):
    def __init__(self, in_dim: int, num_prototypes: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.prototypes = nn.Linear(in_dim, num_prototypes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.prototypes(x)


def list_preprocess_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p.parent for p in root.rglob("preprocess_meta.json"))


def load_slide_id(preprocess_dir: Path) -> str:
    meta_path = preprocess_dir / "preprocess_meta.json"
    if not meta_path.exists():
        return preprocess_dir.name
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return preprocess_dir.name
    slide_id = meta.get("slide_id")
    return str(slide_id) if slide_id else preprocess_dir.name


def load_roi_data(
    preprocess_dir: Path,
    features_path: Path,
    roi_a: ROICandidate,
    roi_b: ROICandidate,
    roi_size: int,
    tile_min_tissue_for_valid: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import zarr

    qc_root = zarr.open(str(preprocess_dir / "qc_grids.zarr"), mode="r")
    tissue_arr = qc_root["tissue_frac"]
    root = zarr.open(str(features_path), mode="r")
    features_arr = root["features"] if "features" in root else root
    r0, c0 = int(roi_a.grid_i), int(roi_a.grid_j)
    r1, c1 = int(roi_b.grid_i), int(roi_b.grid_j)
    xa = np.asarray(features_arr[r0 : r0 + roi_size, c0 : c0 + roi_size, :], dtype=np.float32)
    xb = np.asarray(features_arr[r1 : r1 + roi_size, c1 : c1 + roi_size, :], dtype=np.float32)
    ta = np.asarray(tissue_arr[r0 : r0 + roi_size, c0 : c0 + roi_size], dtype=np.float32)
    tb = np.asarray(tissue_arr[r1 : r1 + roi_size, c1 : c1 + roi_size], dtype=np.float32)
    valid_a = ta >= float(tile_min_tissue_for_valid)
    valid_b = tb >= float(tile_min_tissue_for_valid)
    return xa, xb, valid_a, valid_b


def _random_crop(
    features: np.ndarray,
    valid_mask: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    h, w = features.shape[:2]
    if crop_size > h or crop_size > w:
        raise ValueError("Crop size larger than ROI grid.")
    max_r = h - crop_size
    max_c = w - crop_size
    r0 = int(rng.integers(0, max_r + 1))
    c0 = int(rng.integers(0, max_c + 1))
    return _crop_at(features, valid_mask, r0, c0, crop_size)


def _crop_at(
    features: np.ndarray,
    valid_mask: np.ndarray,
    r0: int,
    c0: int,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    h, w = features.shape[:2]
    if r0 < 0 or c0 < 0 or r0 + crop_size > h or c0 + crop_size > w:
        raise ValueError("Crop location out of bounds.")
    crop = features[r0 : r0 + crop_size, c0 : c0 + crop_size, :]
    valid = valid_mask[r0 : r0 + crop_size, c0 : c0 + crop_size]
    return crop, valid, (r0, c0, crop_size)


def _global_crop_positions(
    h: int,
    w: int,
    crop_size: int,
    num_global: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    if num_global != 2:
        max_r = h - crop_size
        max_c = w - crop_size
        return [
            (int(rng.integers(0, max_r + 1)), int(rng.integers(0, max_c + 1)))
            for _ in range(int(num_global))
        ]
    max_r = h - crop_size
    max_c = w - crop_size
    if max_r < 0 or max_c < 0:
        return []
    if crop_size >= max_r and crop_size >= max_c:
        if rng.random() < 0.5:
            return [(0, 0), (max_r, max_c)]
        return [(0, max_c), (max_r, 0)]
    return [
        (int(rng.integers(0, max_r + 1)), int(rng.integers(0, max_c + 1)))
        for _ in range(int(num_global))
    ]


def _apply_augmentations(
    crop: np.ndarray,
    valid_mask: np.ndarray,
    rng: np.random.Generator,
    flip_prob: float,
    posterize_step: float,
    posterize_noise_std: float,
    posterize_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    out = crop
    mask = valid_mask
    if rng.random() < flip_prob:
        out = np.flip(out, axis=1)
        mask = np.flip(mask, axis=1)
    if rng.random() < flip_prob:
        out = np.flip(out, axis=0)
        mask = np.flip(mask, axis=0)
    if rng.random() < posterize_prob:
        if posterize_step > 0:
            out = np.round(out / float(posterize_step)) * float(posterize_step)
        if posterize_noise_std > 0:
            out = out + rng.normal(0.0, float(posterize_noise_std), size=out.shape).astype(out.dtype, copy=False)
    out = np.where(mask[..., None], out, 0.0)
    return out, mask


def _make_mask(valid_mask: np.ndarray, rng: np.random.Generator, ratio: float) -> np.ndarray:
    if ratio <= 0:
        return np.zeros(valid_mask.shape, dtype=bool)
    mask = rng.random(valid_mask.shape) < float(ratio)
    return mask & valid_mask


def build_views_for_roi(
    features: np.ndarray,
    valid_mask: np.ndarray,
    rng: np.random.Generator,
    config: ViewConfig,
    *,
    roi_id: str,
) -> Tuple[List[CropView], List[CropView]]:
    student_views: List[CropView] = []
    teacher_views: List[CropView] = []
    h, w = features.shape[:2]
    global_positions = _global_crop_positions(
        h,
        w,
        int(config.global_crop_size),
        int(config.num_global),
        rng,
    )
    for r0, c0 in global_positions:
        crop, valid, bbox = _crop_at(
            features,
            valid_mask,
            int(r0),
            int(c0),
            int(config.global_crop_size),
        )
        crop, valid = _apply_augmentations(
            crop,
            valid,
            rng,
            float(config.flip_prob),
            float(config.posterize_step),
            float(config.posterize_noise_std),
            float(config.posterize_prob),
        )
        teacher_views.append(
            CropView(
                roi_id=roi_id,
                view_type="global",
                features=crop,
                valid_mask=valid,
                mask=_make_mask(valid, rng, float(config.teacher_mask_ratio_global)),
                bbox=bbox,
            )
        )
        student_views.append(
            CropView(
                roi_id=roi_id,
                view_type="global",
                features=crop,
                valid_mask=valid,
                mask=_make_mask(valid, rng, float(config.student_mask_ratio_global)),
                bbox=bbox,
            )
        )
    for _ in range(int(config.num_local)):
        crop, valid, bbox = _random_crop(features, valid_mask, int(config.local_crop_size), rng)
        crop, valid = _apply_augmentations(
            crop,
            valid,
            rng,
            float(config.flip_prob),
            float(config.posterize_step),
            float(config.posterize_noise_std),
            float(config.posterize_prob),
        )
        student_views.append(
            CropView(
                roi_id=roi_id,
                view_type="local",
                features=crop,
                valid_mask=valid,
                mask=_make_mask(valid, rng, float(config.student_mask_ratio_local)),
                bbox=bbox,
            )
        )
    return teacher_views, student_views


def _teacher_targets(
    model: SlideViTiBOT,
    views: List[CropView],
    device: torch.device,
    roi_size: int,
    teacher_temp: float,
    center_momentum: float,
) -> Tuple[
    Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    Dict[Tuple[str, int, int, int], torch.Tensor],
]:
    targets_by_roi: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    cls_map: Dict[Tuple[str, int, int, int], torch.Tensor] = {}
    token_center_sum = torch.zeros_like(model.center_token)
    cls_center_sum = torch.zeros_like(model.center_cls)
    token_center_count = 0
    cls_center_count = 0
    num_prototypes = int(model.center_token.numel())

    for view in views:
        feats = torch.from_numpy(view.features).unsqueeze(0).to(device)
        valid = torch.from_numpy(view.valid_mask).unsqueeze(0).to(device)
        token_logits, cls_logits = model.forward_teacher(feats, valid)
        token_logits = token_logits.squeeze(0)
        cls_logits = cls_logits.squeeze(0)
        valid_flat = torch.from_numpy(view.valid_mask.reshape(-1)).to(device)
        if valid_flat.any():
            token_center_sum = token_center_sum + token_logits[valid_flat].mean(dim=0)
            token_center_count += 1
        cls_center_sum = cls_center_sum + cls_logits
        cls_center_count += 1

        token_probs = F.softmax((token_logits - model.center_token) / float(teacher_temp), dim=-1)
        cls_prob = F.softmax((cls_logits - model.center_cls) / float(teacher_temp), dim=-1)
        cls_map[(view.roi_id, *view.bbox)] = cls_prob

        if view.roi_id not in targets_by_roi:
            sum_probs = torch.zeros((roi_size * roi_size, num_prototypes), device=device)
            count = torch.zeros((roi_size * roi_size,), device=device)
            targets_by_roi[view.roi_id] = (sum_probs, count)
        else:
            sum_probs, count = targets_by_roi[view.roi_id]

        h, w = view.valid_mask.shape
        r0, c0, _ = view.bbox
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        flat_idx = (grid_r.reshape(-1) + int(r0)) * int(roi_size) + (grid_c.reshape(-1) + int(c0))
        valid_flat = torch.from_numpy(view.valid_mask.reshape(-1)).to(device)
        if valid_flat.any():
            flat_idx = flat_idx[valid_flat]
            token_probs_valid = token_probs[valid_flat]
            sum_probs.index_add_(0, flat_idx, token_probs_valid)
            count.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=count.dtype, device=device))

    if token_center_count > 0:
        token_center_mean = token_center_sum / float(token_center_count)
        model.center_token.mul_(center_momentum).add_(token_center_mean, alpha=1.0 - center_momentum)
    if cls_center_count > 0:
        cls_center_mean = cls_center_sum / float(cls_center_count)
        model.center_cls.mul_(center_momentum).add_(cls_center_mean, alpha=1.0 - center_momentum)

    targets_final: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for roi_id, (sum_probs, count) in targets_by_roi.items():
        count_safe = count.clamp_min(1.0)
        targets_final[roi_id] = (sum_probs / count_safe.unsqueeze(-1), count)
    return targets_final, cls_map


def ibot_loss(
    model: SlideViTiBOT,
    teacher_views: List[CropView],
    student_views: List[CropView],
    device: torch.device,
    cls_weight: float,
    student_temp: float,
    teacher_temp: float,
    center_momentum: float,
    roi_size: int,
) -> torch.Tensor:
    targets_by_roi, cls_map = _teacher_targets(
        model,
        teacher_views,
        device,
        roi_size,
        teacher_temp,
        center_momentum,
    )
    loss_sum = torch.tensor(0.0, device=device)
    count = 0
    cls_loss = torch.tensor(0.0, device=device)
    cls_count = 0
    for view in student_views:
        feats = torch.from_numpy(view.features).unsqueeze(0).to(device)
        valid = torch.from_numpy(view.valid_mask).unsqueeze(0).to(device)
        mask = torch.from_numpy(view.mask).unsqueeze(0).to(device)
        token_logits, cls_logits = model.forward_student(feats, valid, mask)
        token_logits = token_logits.squeeze(0)
        cls_logits = cls_logits.squeeze(0)
        targets_entry = targets_by_roi.get(view.roi_id)
        if targets_entry is None:
            continue
        target_probs, target_count = targets_entry
        h, w = view.valid_mask.shape
        r0, c0, _ = view.bbox
        mask_flat = torch.from_numpy(view.mask.reshape(-1)).to(device)
        if mask_flat.any():
            mask_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
            rows = (mask_idx // int(w)).to(device)
            cols = (mask_idx % int(w)).to(device)
            roi_idx = (rows + int(r0)) * int(roi_size) + (cols + int(c0))
            valid_target = target_count[roi_idx] > 0
            if valid_target.any():
                sel_mask_idx = mask_idx[valid_target]
                sel_roi_idx = roi_idx[valid_target]
                log_q = F.log_softmax(token_logits[sel_mask_idx] / float(student_temp), dim=-1)
                target = target_probs[sel_roi_idx]
                loss_sum = loss_sum + torch.sum(target * (torch.log(target + 1e-6) - log_q))
                count += int(sel_mask_idx.numel())
        if cls_weight > 0 and view.view_type == "global":
            cls_target = cls_map.get((view.roi_id, *view.bbox))
            if cls_target is not None:
                log_q = F.log_softmax(cls_logits / float(student_temp), dim=-1)
                cls_loss = cls_loss + torch.sum(cls_target * (torch.log(cls_target + 1e-6) - log_q))
                cls_count += 1
    if count == 0:
        return torch.tensor(0.0, device=device)
    loss = loss_sum / float(count)
    if cls_weight > 0 and cls_count > 0:
        loss = loss + (cls_weight * (cls_loss / float(cls_count)))
    return loss


def ema_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    t = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return start + (end - start) * t


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    if warmup_steps < 1:
        warmup_steps = 1
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def run_training(
    preprocess_dirs: List[Path],
    *,
    epochs: int,
    steps_per_epoch: Optional[int],
    batch_size: int,
    roi_size: int,
    target_passed: int,
    min_tissue_frac: float,
    max_artifact_frac: float,
    tile_min_tissue_for_valid: float,
    min_valid_frac: float,
    group_cover_min: float,
    tile_min_tissue_for_group: float,
    merge_radius_tiles: int,
    min_group_tiles_frac: float,
    max_artifact_frac_for_group: float,
    device: str,
    encode_batch_size: int,
    overwrite_features: bool,
    lr: float,
    weight_decay: float,
    warmup_frac: float,
    base_seed: Optional[int],
    cls_weight: float,
    student_mask_ratio_global: float,
    student_mask_ratio_local: float,
    student_temp: float,
    teacher_temp: float,
    center_momentum: float,
    num_prototypes: int,
    head_hidden_dim: int,
) -> None:
    if not preprocess_dirs:
        raise ValueError("No preprocess directories provided.")

    rng = np.random.default_rng(base_seed)
    view_config = ViewConfig(
        roi_size=int(roi_size),
        student_mask_ratio_global=float(student_mask_ratio_global),
        student_mask_ratio_local=float(student_mask_ratio_local),
    )
    model_config = SlideViTConfig(
        input_dim=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_act="swiglu",
        norm_type="rmsnorm",
    )
    device_t = torch.device(device)
    model = SlideViTiBOT(
        model_config,
        num_prototypes=int(num_prototypes),
        head_hidden_dim=int(head_hidden_dim) if head_hidden_dim > 0 else None,
    ).to(device_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    model.student.train()
    model.teacher.eval()

    total_steps = steps_per_epoch
    if total_steps is None or total_steps <= 0:
        total_steps = max(1, int(np.ceil(len(preprocess_dirs) / float(batch_size))))
    total_train_steps = int(total_steps) * int(epochs)
    warmup_steps = max(1, int(float(warmup_frac) * float(total_train_steps)))
    global_step = 0

    for epoch in range(int(epochs)):
        rng.shuffle(preprocess_dirs)
        idx = 0
        for step in range(int(total_steps)):
            if idx + batch_size > len(preprocess_dirs):
                idx = 0
                rng.shuffle(preprocess_dirs)
            batch_dirs = preprocess_dirs[idx : idx + batch_size]
            idx += batch_size

            optimizer.zero_grad()
            lr_scale = cosine_lr(global_step, total_train_steps, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(lr) * float(lr_scale)
            batch_loss = torch.tensor(0.0, device=device_t)
            batch_count = 0

            for slide_offset, preprocess_dir in enumerate(batch_dirs):
                seed = None
                if base_seed is not None:
                    seed = int(base_seed) + epoch * 100000 + step * 100 + slide_offset
                result = sample_candidate_rois_selective(
                    preprocess_dir=preprocess_dir,
                    qc_zarr_path=preprocess_dir / "qc_grids.zarr",
                    features_path=preprocess_dir / "features.zarr",
                    roi_size=int(roi_size),
                    target_passed=int(target_passed),
                    min_tissue_frac=float(min_tissue_frac),
                    max_artifact_frac=float(max_artifact_frac),
                    tile_min_tissue_for_valid=float(tile_min_tissue_for_valid),
                    min_valid_frac=float(min_valid_frac),
                    group_cover_min=float(group_cover_min),
                    tile_min_tissue_for_group=float(tile_min_tissue_for_group),
                    merge_radius_tiles=int(merge_radius_tiles),
                    min_group_tiles_frac=float(min_group_tiles_frac),
                    max_artifact_frac_for_group=float(max_artifact_frac_for_group),
                    k=2,
                    device=str(device),
                    batch_size=int(encode_batch_size),
                    overwrite_features=bool(overwrite_features),
                    seed=seed,
                )

                pair = pick_roi_pair(result)
                if pair is None:
                    continue
                roi_a, roi_b = pair
                features_path = Path(result.metadata.get("features_path", preprocess_dir / "features.zarr"))
                try:
                    x_a, x_b, valid_a, valid_b = load_roi_data(
                        preprocess_dir,
                        features_path,
                        roi_a,
                        roi_b,
                        int(roi_size),
                        float(tile_min_tissue_for_valid),
                    )
                except Exception:
                    continue

                teacher_a, student_a = build_views_for_roi(x_a, valid_a, rng, view_config, roi_id="A")
                teacher_b, student_b = build_views_for_roi(x_b, valid_b, rng, view_config, roi_id="B")
                teacher_views = teacher_a + teacher_b
                student_views = student_a + student_b

                loss = ibot_loss(
                    model,
                    teacher_views,
                    student_views,
                    device_t,
                    float(cls_weight),
                    float(student_temp),
                    float(teacher_temp),
                    float(center_momentum),
                    int(roi_size),
                )
                if torch.isfinite(loss):
                    batch_loss = batch_loss + loss
                    batch_count += 1

            if batch_count == 0:
                print(f"[epoch {epoch+1} step {step+1}] no batch items collected")
                continue

            batch_loss = batch_loss / float(batch_count)
            batch_loss.backward()
            optimizer.step()

            momentum = ema_momentum(global_step, total_train_steps, 0.996, 0.9999)
            model.ema_update(momentum)
            global_step += 1
            print(
                f"[epoch {epoch+1} step {step+1}] batch={batch_count} loss={batch_loss.item():.6f} ema={momentum:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-first masked iBOT training loop.")
    parser.add_argument(
        "--preprocess-root",
        type=str,
        default=str(REPO_ROOT / "output"),
        help="Root directory containing preprocess outputs",
    )
    parser.add_argument(
        "--preprocess-dirs",
        nargs="*",
        default=None,
        help="Explicit list of preprocess dirs (overrides --preprocess-root)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="Steps per epoch (0 = auto)")
    parser.add_argument("--batch-size", type=int, default=2, help="Slides per batch")
    parser.add_argument("--roi-size", type=int, default=16, help="ROI size in tiles")
    parser.add_argument("--target-passed", type=int, default=64, help="Target passed candidates per slide")
    parser.add_argument("--min-tissue-frac", type=float, default=0.7, help="Min ROI tissue fraction")
    parser.add_argument("--max-artifact-frac", type=float, default=0.05, help="Max ROI artifact fraction")
    parser.add_argument(
        "--tile-min-tissue",
        type=float,
        default=0.3,
        help="Min tile tissue fraction for valid coverage",
    )
    parser.add_argument("--min-valid-frac", type=float, default=0.5, help="Min valid tile coverage fraction")
    parser.add_argument("--group-cover-min", type=float, default=0.8, help="Min group coverage")
    parser.add_argument("--tile-min-tissue-group", type=float, default=0.2, help="Min tile tissue for grouping")
    parser.add_argument("--merge-radius-tiles", type=int, default=4, help="Merge radius (tiles)")
    parser.add_argument("--min-group-tiles-frac", type=float, default=0.3, help="Min group tiles frac")
    parser.add_argument(
        "--max-artifact-frac-for-group",
        type=float,
        default=0.25,
        help="Max artifact frac for group formation",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for encoding/training")
    parser.add_argument("--encode-batch-size", type=int, default=32, help="Encoder batch size")
    parser.add_argument("--overwrite-features", action="store_true", help="Overwrite features.zarr")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup-frac", type=float, default=0.05, help="Warmup fraction for cosine LR")
    parser.add_argument("--base-seed", type=int, default=None, help="Base seed for deterministic sampling")
    parser.add_argument("--cls-weight", type=float, default=0.1, help="Optional CLS distill weight")
    parser.add_argument("--student-temp", type=float, default=0.1, help="Student temperature")
    parser.add_argument("--teacher-temp", type=float, default=0.07, help="Teacher temperature")
    parser.add_argument("--center-momentum", type=float, default=0.9, help="Centering momentum")
    parser.add_argument("--num-prototypes", type=int, default=8192, help="Number of iBOT prototypes")
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=0,
        help="Hidden dim for iBOT head (0 = use embed dim)",
    )
    parser.add_argument(
        "--student-mask-global",
        type=float,
        default=0.45,
        help="Student mask ratio for global crops",
    )
    parser.add_argument(
        "--student-mask-local",
        type=float,
        default=0.55,
        help="Student mask ratio for local crops",
    )
    args = parser.parse_args()

    preprocess_dirs: List[Path]
    if args.preprocess_dirs:
        preprocess_dirs = [Path(p) for p in args.preprocess_dirs]
    else:
        preprocess_dirs = list_preprocess_dirs(Path(args.preprocess_root))

    run_training(
        preprocess_dirs,
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch) if args.steps_per_epoch else None,
        batch_size=int(args.batch_size),
        roi_size=int(args.roi_size),
        target_passed=int(args.target_passed),
        min_tissue_frac=float(args.min_tissue_frac),
        max_artifact_frac=float(args.max_artifact_frac),
        tile_min_tissue_for_valid=float(args.tile_min_tissue),
        min_valid_frac=float(args.min_valid_frac),
        group_cover_min=float(args.group_cover_min),
        tile_min_tissue_for_group=float(args.tile_min_tissue_group),
        merge_radius_tiles=int(args.merge_radius_tiles),
        min_group_tiles_frac=float(args.min_group_tiles_frac),
        max_artifact_frac_for_group=float(args.max_artifact_frac_for_group),
        device=str(args.device),
        encode_batch_size=int(args.encode_batch_size),
        overwrite_features=bool(args.overwrite_features),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_frac=float(args.warmup_frac),
        base_seed=args.base_seed,
        cls_weight=float(args.cls_weight),
        student_mask_ratio_global=float(args.student_mask_global),
        student_mask_ratio_local=float(args.student_mask_local),
        student_temp=float(args.student_temp),
        teacher_temp=float(args.teacher_temp),
        center_momentum=float(args.center_momentum),
        num_prototypes=int(args.num_prototypes),
        head_hidden_dim=int(args.head_hidden_dim),
    )


if __name__ == "__main__":
    main()
