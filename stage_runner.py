from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.dataloader import SlideDataset, create_dataloader
from pipeline.roi_sampler import ROICandidate, sample_candidate_rois_selective
from pipeline.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]


def _serialize_candidate(candidate: ROICandidate) -> Dict[str, Any]:
    return {
        "roi_id": int(candidate.roi_id),
        "grid_i": int(candidate.grid_i),
        "grid_j": int(candidate.grid_j),
        "tissue_frac": float(candidate.tissue_frac),
        "artifact_frac": float(candidate.artifact_frac),
        "group_id": int(candidate.group_id) if candidate.group_id is not None else None,
    }


def preprocess_stage(
    slides_root: Path,
    output_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
    preprocess_kwargs: Optional[dict],
) -> None:
    dataset = SlideDataset(
        slides_root=slides_root,
        output_root=output_root,
        preprocess_on_access=True,
        preprocess_kwargs=preprocess_kwargs,
    )
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total = len(dataset)
    processed = 0
    for batch in loader:
        processed += len(batch)
        print(f"[preprocess] {processed}/{total} slides processed")


def sample_stage(
    preprocess_root: Path,
    *,
    batch_size: int,
    num_workers: int,
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
    seed: Optional[int],
    overwrite_samples: bool,
) -> None:
    dataset = SlideDataset(preprocess_root=preprocess_root)
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total = len(dataset)
    processed = 0
    for batch in loader:
        for record in batch:
            sample_path = record.preprocess_dir / "roi_samples.json"
            if sample_path.exists() and not overwrite_samples:
                continue
            result = sample_candidate_rois_selective(
                preprocess_dir=record.preprocess_dir,
                qc_zarr_path=record.qc_zarr_path,
                features_path=record.features_path,
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
            payload = {
                "slide_id": record.slide_id,
                "roi_size": int(roi_size),
                "passed_candidates": [_serialize_candidate(c) for c in result.passed_candidates],
                "selected_candidates": [
                    _serialize_candidate(c) for c in (result.selected_candidates or [])
                ],
                "metadata": result.metadata,
            }
            sample_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        processed += len(batch)
        print(f"[sample] {processed}/{total} slides sampled")


def train_stage(
    preprocess_root: Path,
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
    preprocess_dirs = sorted(p.parent for p in preprocess_root.rglob("preprocess_meta.json"))
    run_training(
        preprocess_dirs,
        epochs=int(epochs),
        steps_per_epoch=steps_per_epoch,
        batch_size=int(batch_size),
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
        device=str(device),
        encode_batch_size=int(encode_batch_size),
        overwrite_features=bool(overwrite_features),
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_frac=float(warmup_frac),
        base_seed=base_seed,
        cls_weight=float(cls_weight),
        student_mask_ratio_global=float(student_mask_ratio_global),
        student_mask_ratio_local=float(student_mask_ratio_local),
        student_temp=float(student_temp),
        teacher_temp=float(teacher_temp),
        center_momentum=float(center_momentum),
        num_prototypes=int(num_prototypes),
        head_hidden_dim=int(head_hidden_dim),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline stage runner (preprocess -> sample -> train).")
    parser.add_argument("--stage", choices=["preprocess", "sample", "train", "all"], default="all")
    parser.add_argument("--slides-root", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--manifest", type=str, default=None, help="Parquet manifest with SeriesInstanceUID etc.")
    parser.add_argument("--tmp-root", type=str, default="/tmp/idc_wsi", help="Temp download root")
    parser.add_argument("--preprocess-root", type=str, default=str(REPO_ROOT / "output"))
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--preprocess-kwargs", type=str, default=None, help="JSON string for preprocess_wsi args")
    parser.add_argument("--overwrite-samples", action="store_true")

    parser.add_argument("--roi-size", type=int, default=16)
    parser.add_argument("--target-passed", type=int, default=64)
    parser.add_argument("--min-tissue-frac", type=float, default=0.7)
    parser.add_argument("--max-artifact-frac", type=float, default=0.05)
    parser.add_argument("--tile-min-tissue", type=float, default=0.3)
    parser.add_argument("--min-valid-frac", type=float, default=0.5)
    parser.add_argument("--group-cover-min", type=float, default=0.8)
    parser.add_argument("--tile-min-tissue-group", type=float, default=0.2)
    parser.add_argument("--merge-radius-tiles", type=int, default=4)
    parser.add_argument("--min-group-tiles-frac", type=float, default=0.3)
    parser.add_argument("--max-artifact-frac-for-group", type=float, default=0.25)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--cls-weight", type=float, default=0.1)
    parser.add_argument("--student-mask-global", type=float, default=0.45)
    parser.add_argument("--student-mask-local", type=float, default=0.55)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--teacher-temp", type=float, default=0.07)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--num-prototypes", type=int, default=8192)
    parser.add_argument("--head-hidden-dim", type=int, default=0)

    args = parser.parse_args()

    slides_root = Path(args.slides_root)
    preprocess_root = Path(args.preprocess_root)
    output_root = Path(args.output_root) if args.output_root else preprocess_root

    preprocess_kwargs = None
    if args.preprocess_kwargs:
        preprocess_kwargs = json.loads(args.preprocess_kwargs)

    if args.stage in ("preprocess", "all"):
        if args.manifest:
            import pandas as pd
            from pipeline.preprocess_qc import preprocess_wsi
            from pipeline.idc_materialize import materialize_series_to_tmp

            df = pd.read_parquet(args.manifest)
            out_root = output_root
            tmp_root = Path(args.tmp_root)

            for r in df.itertuples(index=False):
                series_uid = getattr(r, "SeriesInstanceUID")
                slide_id = series_uid  # stable ID
                out_dir = out_root / f"{slide_id}_qc"

                if (out_dir / "preprocess_meta.json").exists():
                    continue

                extra_metadata: Dict[str, Any] = {"SeriesInstanceUID": str(series_uid)}
                for key in ("series_aws_url", "collection_id", "PatientID"):
                    if hasattr(r, key):
                        value = getattr(r, key)
                        if not pd.isna(value):
                            extra_metadata[key] = value

                with materialize_series_to_tmp(series_uid, tmp_root) as series_path:
                    # series_path is a directory; preprocess_qc now supports it
                    preprocess_wsi(
                        series_path,
                        output_dir=out_dir,
                        target_mag=20.0,
                        min_tissue_frac=0.5,
                        device=str(args.device),
                        extra_metadata=extra_metadata,
                    )
            print("[preprocess] done (manifest mode)")
        else:
            preprocess_stage(...)


    if args.stage in ("sample", "all"):
        sample_stage(
            output_root,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
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
            seed=args.seed,
            overwrite_samples=bool(args.overwrite_samples),
        )

    if args.stage in ("train", "all"):
        train_stage(
            output_root,
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
