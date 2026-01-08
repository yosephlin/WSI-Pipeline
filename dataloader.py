from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_SLIDE_EXTS: Tuple[str, ...] = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".dcm",
)


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    preprocess_dir: Path
    qc_zarr_path: Path
    features_path: Path
    slide_path: Optional[Path] = None


def _discover_preprocess_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p.parent for p in root.rglob("preprocess_meta.json"))


def _discover_slide_paths(root: Path, exts: Sequence[str]) -> List[Path]:
    if not root.exists():
        return []
    exts_lower = {e.lower() for e in exts}
    slides: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts_lower:
            slides.append(path)
    return sorted(slides)


def _default_preprocess_dir(slide_path: Path, output_root: Optional[Path]) -> Path:
    if output_root is None:
        return slide_path.parent / f"{slide_path.stem}_qc"
    return output_root / f"{slide_path.stem}_qc"


class SlideDataset(Dataset[SlideRecord]):
    """
    Dataset that yields slide records for ROI sampling or preprocessing.

    When preprocess_on_access is True, missing preprocess outputs are generated
    on demand using preprocess_wsi.
    """

    def __init__(
        self,
        *,
        preprocess_root: Optional[Path] = None,
        slides_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
        slide_exts: Sequence[str] = DEFAULT_SLIDE_EXTS,
        preprocess_on_access: bool = False,
        preprocess_kwargs: Optional[dict] = None,
    ) -> None:
        if preprocess_root is None and slides_root is None:
            raise ValueError("Provide either preprocess_root or slides_root.")

        self.preprocess_on_access = bool(preprocess_on_access)
        self.preprocess_kwargs = preprocess_kwargs or {}
        self.records: List[SlideRecord] = []

        if preprocess_root is not None:
            for preprocess_dir in _discover_preprocess_dirs(Path(preprocess_root)):
                slide_id = preprocess_dir.name.replace("_qc", "")
                self.records.append(
                    SlideRecord(
                        slide_id=slide_id,
                        preprocess_dir=preprocess_dir,
                        qc_zarr_path=preprocess_dir / "qc_grids.zarr",
                        features_path=preprocess_dir / "features.zarr",
                    )
                )
        else:
            slides_root = Path(slides_root)
            output_root = Path(output_root) if output_root is not None else None
            for slide_path in _discover_slide_paths(slides_root, slide_exts):
                preprocess_dir = _default_preprocess_dir(slide_path, output_root)
                slide_id = slide_path.stem
                self.records.append(
                    SlideRecord(
                        slide_id=slide_id,
                        preprocess_dir=preprocess_dir,
                        qc_zarr_path=preprocess_dir / "qc_grids.zarr",
                        features_path=preprocess_dir / "features.zarr",
                        slide_path=slide_path,
                    )
                )

        if not self.records:
            raise ValueError("No slides found for dataset.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> SlideRecord:
        record = self.records[idx]
        if self.preprocess_on_access:
            if not record.qc_zarr_path.exists():
                if record.slide_path is None:
                    raise FileNotFoundError(f"Missing qc_grids.zarr at {record.qc_zarr_path}")
                from pipeline.preprocess_qc import preprocess_wsi

                preprocess_wsi(
                    record.slide_path,
                    output_dir=record.preprocess_dir,
                    **self.preprocess_kwargs,
                )
        return record


def collate_slides(batch: Iterable[SlideRecord]) -> List[SlideRecord]:
    return list(batch)


def create_dataloader(
    dataset: SlideDataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_slides,
        prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
        persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
        pin_memory=bool(pin_memory),
    )

