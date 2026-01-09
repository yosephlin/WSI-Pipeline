"""
Unified WSI preprocessing with tissue segmentation and GrandQC artifact detection.

This module provides a single entry point that:
  1. Computes Otsu-based tissue mask on a low-magnification thumbnail
  2. Runs GrandQC artifact segmentation within tissue regions
  3. Generates a tile grid (512x512) constrained to tissue bounding box
  4. Stores per-tile QC channels in Zarr (fast 2D window queries) + Parquet manifest (analytics)

Usage:
    from pipeline.preprocess_qc import preprocess_wsi

    result = preprocess_wsi(
        "path/to/slide.svs",
        output_dir="path/to/output",
        target_mag=20.0,  # 20x = 0.5 µm/px
        min_tissue_frac=0.5,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError as e:
    cv2 = None
    _CV2_IMPORT_ERROR = e
else:
    _CV2_IMPORT_ERROR = None


__all__ = [
    "PreprocessResult",
    "preprocess_wsi",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_SIZE_PX = 512
TARGET_MAG = 20.0  # 20x = 0.5 µm/px
MIN_TISSUE_FRAC = 0.5
GRANDQC_INPUT_SIZE = 512
GRANDQC_BATCH_SIZE = 4
GRANDQC_MPP = 1.5


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _open_wsi(path: Path):
    # OpenSlide for classic formats; WsiDicom for DICOM WSI folders
    if path.is_dir() or path.suffix.lower() == ".dcm":
        from wsidicom import WsiDicom
        w = WsiDicom.open(path if path.is_dir() else path.parent)
        return _WsiDicomAsOpenSlide(w)
    else:
        import openslide
        return openslide.OpenSlide(str(path))


class _WsiDicomAsOpenSlide:
    """Minimal OpenSlide-like wrapper around wsidicom, using level-0 coordinates."""
    def __init__(self, wsi):
        self._wsi = wsi
        self.level_dimensions = [
            (int(l.datasets[0].TotalPixelMatrixColumns), int(l.datasets[0].TotalPixelMatrixRows))
            for l in wsi.levels
        ]
        # wsidicom levels expose pyramid index as `level.level` (downsample = 2**level.level)
        self.level_downsamples = [float(2 ** int(l.level)) for l in wsi.levels]
        self.properties = {}  # optionally populate if you want

    def read_region(self, location, level, size):
        x0, y0 = location  # OpenSlide style: level-0 coords
        ds = float(self.level_downsamples[level])
        # wsidicom expects coords relative to the chosen level
        xl, yl = int(round(x0 / ds)), int(round(y0 / ds))
        img = self._wsi.read_region((xl, yl), level, size)
        return img  # PIL.Image

    def close(self):
        self._wsi.close()

DEFAULT_GRANDQC_MODEL = _repo_root() / "pipeline" / "models" / "GrandQC_MPP15.pth"


def _require_cv2() -> None:
    if _CV2_IMPORT_ERROR is not None:
        raise ImportError("OpenCV (`cv2`) is required.") from _CV2_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Thumbnail and tissue mask
# ---------------------------------------------------------------------------


def _estimate_level0_mpp(slide: Any) -> Optional[float]:
    props = getattr(slide, "properties", None) or {}
    candidates = []
    for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
        try:
            v = props.get(key)
            if v is not None:
                candidates.append(float(v))
        except Exception:
            continue
    return float(sum(candidates) / len(candidates)) if candidates else None


def _estimate_objective_power(slide: Any, level0_mpp: Optional[float]) -> Optional[float]:
    props = getattr(slide, "properties", None) or {}
    for key in ("openslide.objective-power", "aperio.AppMag", "hamamatsu.SourceLens"):
        try:
            v = props.get(key)
            if v is not None:
                f = float(v)
                if f > 0:
                    return f
        except Exception:
            continue
    if level0_mpp is not None and level0_mpp > 0:
        return 10.0 / level0_mpp
    return None


@dataclass(frozen=True)
class Thumbnail:
    rgb: np.ndarray  # uint8 [H, W, 3]
    level: int
    downsample: float
    level_dimensions: Tuple[int, int]  # (W, H)
    slide_dimensions: Tuple[int, int]  # (W0, H0)
    scale_x: float
    scale_y: float
    level0_mpp: Optional[float]
    objective_power: Optional[float]
    magnification: Optional[float]


def _build_thumbnail(
    slide: Any,
    *,
    target_magnification: Optional[float] = None,
    max_dim: int = 2048,
) -> Thumbnail:
    slide_dims = tuple(int(v) for v in slide.level_dimensions[0])
    level0_mpp = _estimate_level0_mpp(slide)
    objective_power = _estimate_objective_power(slide, level0_mpp)

    level_dims = [tuple(int(v) for v in d) for d in slide.level_dimensions]
    ds_list = getattr(slide, "level_downsamples", [1.0] * len(level_dims))

    if target_magnification is not None and objective_power is not None:
        ds_target = objective_power / target_magnification
        ds_min = max(slide_dims) / max_dim if max_dim > 0 else ds_target
        desired_ds = max(ds_target, ds_min)

        best_lvl = 0
        best_err = float("inf")
        for lvl, ds in enumerate(ds_list):
            err = abs(np.log(float(ds) / desired_ds))
            if err < best_err:
                best_err = err
                best_lvl = lvl
    else:
        best_lvl = 0
        best_err = float("inf")
        for lvl, (w, h) in enumerate(level_dims):
            mx = max(w, h)
            if mx <= max_dim:
                err = abs(mx - max_dim)
                if err < best_err:
                    best_err = err
                    best_lvl = lvl
        if best_err == float("inf"):
            best_lvl = len(level_dims) - 1

    w_read, h_read = level_dims[best_lvl]
    region = slide.read_region((0, 0), best_lvl, (w_read, h_read)).convert("RGB")
    rgb = np.asarray(region, dtype=np.uint8)

    ds = float(ds_list[best_lvl]) if best_lvl < len(ds_list) else 1.0
    scale_x = w_read / slide_dims[0] if slide_dims[0] > 0 else 1.0
    scale_y = h_read / slide_dims[1] if slide_dims[1] > 0 else 1.0
    mag = (objective_power / ds) if objective_power and ds > 0 else None

    return Thumbnail(
        rgb=rgb,
        level=best_lvl,
        downsample=ds,
        level_dimensions=(w_read, h_read),
        slide_dimensions=slide_dims,
        scale_x=scale_x,
        scale_y=scale_y,
        level0_mpp=level0_mpp,
        objective_power=objective_power,
        magnification=mag,
    )


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    _require_cv2()
    h, w = mask_u8.shape
    inv = cv2.bitwise_not(mask_u8)
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, flood_inv)


def _remove_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    _require_cv2()
    if min_area <= 0:
        return mask_u8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask_u8
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels].astype(np.uint8) * 255)


def _otsu_tissue_mask(
    rgb: np.ndarray,
    *,
    min_component_area: int = 512,
    morph_kernel: int = 7,
) -> np.ndarray:
    _require_cv2()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got {rgb.shape}")

    img = rgb.astype(np.uint8, copy=False)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    gray_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sat_blur = cv2.GaussianBlur(sat, (0, 0), sigmaX=1.0)

    thr_gray, _ = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_sat, _ = cv2.threshold(sat_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tissue_by_intensity = gray_blur < thr_gray
    tissue_by_saturation = sat_blur > thr_sat
    tissue = tissue_by_intensity | tissue_by_saturation

    mask_u8 = tissue.astype(np.uint8) * 255
    k = max(3, morph_kernel)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = _fill_holes(mask_u8)
    mask_u8 = _remove_small_components(mask_u8, min_component_area)

    return mask_u8 > 0


def _tissue_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------------------------------------------------------------------------
# GrandQC artifact detection (integrated from artifact_mask.py)
# ---------------------------------------------------------------------------


def _load_grandqc_model(
    model_path: Path,
    device: str = "cuda",
) -> Tuple[Any, Callable]:
    """Load GrandQC model and preprocessing function."""
    from pipeline.artifact_mask import load_grandqc_artifact_model
    return load_grandqc_artifact_model(model_path, device=device)


def _run_grandqc_on_slide(
    slide: Any,
    tissue_mask: np.ndarray,
    model: Any,
    preprocessing_fn: Callable,
    device: str = "cuda",
    batch_size: int = GRANDQC_BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run GrandQC artifact segmentation on slide within tissue regions."""
    from pipeline.artifact_mask import segment_grandqc_artifacts_slide

    return segment_grandqc_artifacts_slide(
        slide,
        tissue_mask,
        model=model,
        preprocessing_fn=preprocessing_fn,
        device=device,
        artifact_mpp_model=GRANDQC_MPP,
        patch_size=GRANDQC_INPUT_SIZE,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Tile grid generation with per-tile QC scores
# ---------------------------------------------------------------------------


@dataclass
class TileRecord:
    """Per-tile information stored in manifest."""
    tile_id: int
    grid_i: int  # row index in tile grid
    grid_j: int  # col index in tile grid
    x0_lvl0: int
    y0_lvl0: int
    x1_lvl0: int
    y1_lvl0: int
    tissue_frac: float
    artifact_frac: float
    rejected: bool = False
    reject_reason: Optional[str] = None


@dataclass
class PreprocessResult:
    """Result of preprocessing a WSI."""
    wsi_path: Path
    output_dir: Path
    thumbnail: Thumbnail
    tissue_mask: np.ndarray
    artifact_mask: np.ndarray
    # Tile grid arrays (H_tiles, W_tiles)
    grid_shape: Tuple[int, int]  # (n_rows, n_cols)
    grid_origin_lvl0: Tuple[int, int]  # (x0, y0) of grid origin in level-0 coords
    tile_size_lvl0: int
    # Per-tile quality maps in grid coordinates
    tissue_frac_grid: np.ndarray  # float16 (H_tiles, W_tiles)
    artifact_frac_grid: np.ndarray  # float16 (H_tiles, W_tiles)
    # Tile records for Parquet
    tiles: List[TileRecord]
    rejected_tiles: List[TileRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _mask_fraction_in_bbox(
    mask: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
    scale_x: float, scale_y: float,
) -> float:
    """Compute fraction of True pixels in mask for a level-0 bbox."""
    h, w = mask.shape[:2]
    mx0 = int(np.floor(x0 * scale_x))
    my0 = int(np.floor(y0 * scale_y))
    mx1 = int(np.ceil(x1 * scale_x))
    my1 = int(np.ceil(y1 * scale_y))
    mx0, mx1 = max(0, min(mx0, w)), max(0, min(mx1, w))
    my0, my1 = max(0, min(my0, h)), max(0, min(my1, h))
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    region = mask[my0:my1, mx0:mx1]
    return float(region.mean()) if region.size > 0 else 0.0


def _generate_tile_grid_with_qc(
    slide: Any,
    thumbnail: Thumbnail,
    tissue_mask: np.ndarray,
    artifact_mask: np.ndarray,
    *,
    tile_size_lvl0: int,
    min_tissue_frac: float = MIN_TISSUE_FRAC,
    target_mpp: float,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[
    Tuple[int, int],  # grid_shape
    Tuple[int, int],  # grid_origin_lvl0
    np.ndarray,  # tissue_frac_grid
    np.ndarray,  # artifact_frac_grid
    List[TileRecord],  # accepted
    List[TileRecord],  # rejected
]:
    """Generate tile grid with per-tile QC scores computed during tiling."""
    _require_cv2()

    W, H = thumbnail.slide_dimensions
    scale_x, scale_y = thumbnail.scale_x, thumbnail.scale_y

    # Find tissue bounding box
    tissue_bbox = _tissue_bbox(tissue_mask)
    if tissue_bbox is None:
        # No tissue - return empty
        return (0, 0), (0, 0), np.array([]), np.array([]), [], []

    tx0, ty0, tx1, ty1 = tissue_bbox
    # Convert to level-0 and add margin
    sx0 = max(0, int(tx0 / scale_x) - tile_size_lvl0)
    sy0 = max(0, int(ty0 / scale_y) - tile_size_lvl0)
    sx1 = min(W, int(tx1 / scale_x) + tile_size_lvl0)
    sy1 = min(H, int(ty1 / scale_y) + tile_size_lvl0)

    # Align to tile grid
    sx0 = (sx0 // tile_size_lvl0) * tile_size_lvl0
    sy0 = (sy0 // tile_size_lvl0) * tile_size_lvl0

    n_cols = max(1, (sx1 - sx0 + tile_size_lvl0 - 1) // tile_size_lvl0)
    n_rows = max(1, (sy1 - sy0 + tile_size_lvl0 - 1) // tile_size_lvl0)

    # Initialize grid arrays
    tissue_frac_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    artifact_frac_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    total_tiles = n_rows * n_cols
    processed = 0

    # First pass: compute all scores
    for i in range(n_rows):
        for j in range(n_cols):
            x0 = sx0 + j * tile_size_lvl0
            y0 = sy0 + i * tile_size_lvl0
            x1 = min(x0 + tile_size_lvl0, W)
            y1 = min(y0 + tile_size_lvl0, H)

            # Tissue and artifact fractions from thumbnail masks
            tissue_frac = _mask_fraction_in_bbox(tissue_mask, x0, y0, x1, y1, scale_x, scale_y)
            artifact_frac = _mask_fraction_in_bbox(artifact_mask, x0, y0, x1, y1, scale_x, scale_y)

            tissue_frac_grid[i, j] = tissue_frac
            artifact_frac_grid[i, j] = artifact_frac

            processed += 1
            if progress_callback:
                progress_callback(processed, total_tiles)

    # Second pass: apply QC filters
    # Step 1: tissue filter
    tissue_pass = tissue_frac_grid >= min_tissue_frac

    valid_mask_grid = tissue_pass.copy()

    # Build tile records
    tiles: List[TileRecord] = []
    rejected_tiles: List[TileRecord] = []
    tile_id = 0

    for i in range(n_rows):
        for j in range(n_cols):
            x0 = sx0 + j * tile_size_lvl0
            y0 = sy0 + i * tile_size_lvl0
            x1 = min(x0 + tile_size_lvl0, W)
            y1 = min(y0 + tile_size_lvl0, H)

            tissue_frac = float(tissue_frac_grid[i, j])
            artifact_frac = float(artifact_frac_grid[i, j])
            is_valid = bool(valid_mask_grid[i, j])

            # Determine rejection reason
            reject_reason = None
            if not is_valid:
                if tissue_frac < min_tissue_frac:
                    reject_reason = "low_tissue"

            record = TileRecord(
                tile_id=tile_id,
                grid_i=i,
                grid_j=j,
                x0_lvl0=x0,
                y0_lvl0=y0,
                x1_lvl0=x1,
                y1_lvl0=y1,
                tissue_frac=tissue_frac,
                artifact_frac=artifact_frac,
                rejected=not is_valid,
                reject_reason=reject_reason,
            )

            if is_valid:
                tiles.append(record)
            else:
                rejected_tiles.append(record)

            tile_id += 1

    # Convert to float16 for storage efficiency
    tissue_frac_grid = tissue_frac_grid.astype(np.float16)
    artifact_frac_grid = artifact_frac_grid.astype(np.float16)

    return (
        (n_rows, n_cols),
        (sx0, sy0),
        tissue_frac_grid,
        artifact_frac_grid,
        tiles,
        rejected_tiles,
    )


# ---------------------------------------------------------------------------
# Storage: Zarr + Parquet
# ---------------------------------------------------------------------------


def _save_zarr_arrays(
    output_dir: Path,
    grid_shape: Tuple[int, int],
    tissue_frac_grid: np.ndarray,
    artifact_frac_grid: np.ndarray,
    tissue_mask: np.ndarray,
    artifact_mask: np.ndarray,
) -> Path:
    """Save tile-grid-aligned QC channels to Zarr for fast window queries."""
    try:
        import zarr
    except ImportError as e:
        raise ImportError("zarr is required for storage. Install via: pip install zarr") from e

    zarr_path = output_dir / "qc_grids.zarr"
    root = zarr.open(str(zarr_path), mode="w")

    # Tile-grid arrays (for fast 16x16 window queries)
    def _create_array_compat(name: str, data: np.ndarray, chunks: Tuple[int, int]) -> None:
        try:
            root.create_array(
                name,
                data=data,
                chunks=chunks,
            )
            return
        except ValueError as exc:
            if "data parameter was used" not in str(exc):
                raise
        arr = root.create_array(
            name,
            shape=data.shape,
            dtype=data.dtype,
            chunks=chunks,
        )
        arr[:] = data

    _create_array_compat("tissue_frac", tissue_frac_grid, chunks=(64, 64))
    _create_array_compat("artifact_frac", artifact_frac_grid, chunks=(64, 64))

    # Thumbnail-resolution masks
    tissue_mask_u8 = tissue_mask.astype(np.uint8)
    artifact_mask_u8 = artifact_mask.astype(np.uint8)
    _create_array_compat("tissue_mask_thumb", tissue_mask_u8, chunks=(256, 256))
    _create_array_compat("artifact_mask_thumb", artifact_mask_u8, chunks=(256, 256))

    root.attrs["grid_shape"] = list(grid_shape)

    return zarr_path


def _save_parquet_manifest(
    output_dir: Path,
    tiles: List[TileRecord],
    rejected_tiles: List[TileRecord],
    slide_id: str,
) -> Path:
    """Save tile manifest to Parquet for analytics."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("pyarrow is required for Parquet. Install via: pip install pyarrow") from e

    all_tiles = tiles + rejected_tiles

    data = {
        "slide_id": [slide_id] * len(all_tiles),
        "tile_id": [t.tile_id for t in all_tiles],
        "grid_i": [t.grid_i for t in all_tiles],
        "grid_j": [t.grid_j for t in all_tiles],
        "x0_lvl0": [t.x0_lvl0 for t in all_tiles],
        "y0_lvl0": [t.y0_lvl0 for t in all_tiles],
        "x1_lvl0": [t.x1_lvl0 for t in all_tiles],
        "y1_lvl0": [t.y1_lvl0 for t in all_tiles],
        "tissue_frac": [t.tissue_frac for t in all_tiles],
        "artifact_frac": [t.artifact_frac for t in all_tiles],
        "rejected": [t.rejected for t in all_tiles],
        "reject_reason": [t.reject_reason for t in all_tiles],
    }

    table = pa.table(data)
    parquet_path = output_dir / "tile_manifest.parquet"
    pq.write_table(table, str(parquet_path))

    return parquet_path


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------


def preprocess_wsi(
    wsi_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    *,
    tile_size_px: int = TILE_SIZE_PX,
    target_mag: float = TARGET_MAG,
    min_tissue_frac: float = MIN_TISSUE_FRAC,
    grandqc_model_path: Optional[Path] = None,
    grandqc_batch_size: int = GRANDQC_BATCH_SIZE,
    device: str = "cuda",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> PreprocessResult:
    """
    Unified WSI preprocessing pipeline.

    This function runs the complete preprocessing workflow:
      1. Build low-magnification thumbnail
      2. Compute Otsu tissue mask
      3. Run GrandQC artifact segmentation (within tissue regions)
      4. Generate tile grid constrained to tissue bounding box
      5. Apply QC filters (tissue fraction)
      6. Save Zarr arrays (fast window queries) + Parquet manifest (analytics)

    Args:
        wsi_path: Path to WSI file (.svs, .ndpi, .tiff, etc.)
        output_dir: Output directory (default: {wsi_stem}_qc/)
        tile_size_px: Tile size at target magnification (default 512)
        target_mag: Target magnification (default 20x = 0.5 µm/px)
        min_tissue_frac: Minimum tissue fraction to accept tile (default 0.5)
        grandqc_model_path: Path to GrandQC model (default: auto-detect)
        grandqc_batch_size: Batch size for GrandQC inference
        device: Device for GrandQC ("cuda" or "cpu")
        progress_callback: Optional callback(stage, current, total) for progress
        extra_metadata: Optional dict merged into preprocess_meta.json

    Returns:
        PreprocessResult with all computed data and paths

    Example:
        >>> result = preprocess_wsi("slide.svs", target_mag=20.0)
        >>> print(f"Grid shape: {result.grid_shape}")
        >>> print(f"Accepted: {len(result.tiles)}, Rejected: {len(result.rejected_tiles)}")

        # Fast window query during training:
        >>> import zarr
        >>> z = zarr.open(result.output_dir / "qc_grids.zarr")
        >>> tissue_window = z["tissue_frac"][i:i+16, j:j+16]
    """
    try:
        import openslide
    except ImportError as e:
        raise ImportError("OpenSlide is required.") from e

    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    if output_dir is None:
        output_dir = wsi_path.parent / f"{wsi_path.stem}_qc"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide = _open_wsi(wsi_path)
    slide_id = wsi_path.stem

    def _progress(stage: str, current: int, total: int):
        if progress_callback:
            progress_callback(stage, current, total)

    # Step 1: Build thumbnail
    _progress("thumbnail", 0, 1)
    thumbnail = _build_thumbnail(slide, target_magnification=target_mag, max_dim=2048)
    _progress("thumbnail", 1, 1)

    # Step 2: Compute tissue mask
    _progress("tissue_mask", 0, 1)
    tissue_mask = _otsu_tissue_mask(thumbnail.rgb)
    _progress("tissue_mask", 1, 1)

    # Step 3: Run GrandQC artifact segmentation
    _progress("grandqc", 0, 1)
    if grandqc_model_path is None:
        grandqc_model_path = DEFAULT_GRANDQC_MODEL

    if grandqc_model_path.exists():
        model, preprocessing_fn = _load_grandqc_model(grandqc_model_path, device=device)
        artifact_mask, _class_map, grandqc_meta = _run_grandqc_on_slide(
            slide, tissue_mask, model, preprocessing_fn,
            device=device, batch_size=grandqc_batch_size,
        )
        del _class_map  # Not needed, using artifact_mask only
    else:
        # Fallback: no artifact detection
        artifact_mask = np.zeros_like(tissue_mask, dtype=bool)
        grandqc_meta = {"enabled": False, "reason": "model_not_found"}
    _progress("grandqc", 1, 1)

    # Step 4: Compute tile size in level-0 pixels
    level0_mpp = thumbnail.level0_mpp or 0.5
    objective_power = thumbnail.objective_power or 40.0
    target_mpp = (objective_power / target_mag) * level0_mpp
    tile_size_lvl0 = max(1, int(round(tile_size_px * target_mpp / level0_mpp)))

    # Step 5: Generate tile grid with QC scores
    def _tile_progress(current: int, total: int):
        _progress("tiling", current, total)

    (
        grid_shape,
        grid_origin,
        tissue_frac_grid,
        artifact_frac_grid,
        tiles,
        rejected_tiles,
    ) = _generate_tile_grid_with_qc(
        slide,
        thumbnail,
        tissue_mask,
        artifact_mask,
        tile_size_lvl0=tile_size_lvl0,
        min_tissue_frac=min_tissue_frac,
        target_mpp=target_mpp,
        progress_callback=_tile_progress,
    )

    slide.close()

    # Step 6: Save Zarr arrays
    _progress("save_zarr", 0, 1)
    if grid_shape[0] > 0 and grid_shape[1] > 0:
        zarr_path = _save_zarr_arrays(
            output_dir,
            grid_shape,
            tissue_frac_grid,
            artifact_frac_grid,
            tissue_mask,
            artifact_mask,
        )
    else:
        zarr_path = None
    _progress("save_zarr", 1, 1)

    # Step 7: Save Parquet manifest
    _progress("save_parquet", 0, 1)
    parquet_path = _save_parquet_manifest(output_dir, tiles, rejected_tiles, slide_id)
    _progress("save_parquet", 1, 1)

    # Build metadata
    metadata = {
        "wsi_path": str(wsi_path),
        "slide_id": slide_id,
        "slide_dimensions": thumbnail.slide_dimensions,
        "target_magnification": target_mag,
        "target_mpp": target_mpp,
        "tile_size_px": tile_size_px,
        "tile_size_lvl0": tile_size_lvl0,
        "level0_mpp": level0_mpp,
        "grid_shape": grid_shape,
        "grid_origin_lvl0": grid_origin,
        "min_tissue_frac": min_tissue_frac,
        "total_grid_tiles": grid_shape[0] * grid_shape[1] if grid_shape[0] > 0 else 0,
        "accepted_count": len(tiles),
        "rejected_count": len(rejected_tiles),
        "rejected_low_tissue": sum(1 for t in rejected_tiles if t.reject_reason == "low_tissue"),
        "grandqc": grandqc_meta,
        "zarr_path": str(zarr_path) if zarr_path else None,
        "parquet_path": str(parquet_path),
    }
    if extra_metadata:
        for key, value in extra_metadata.items():
            if key not in metadata:
                metadata[key] = value

    # Save metadata JSON
    meta_path = output_dir / "preprocess_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return PreprocessResult(
        wsi_path=wsi_path,
        output_dir=output_dir,
        thumbnail=thumbnail,
        tissue_mask=tissue_mask,
        artifact_mask=artifact_mask,
        grid_shape=grid_shape,
        grid_origin_lvl0=grid_origin,
        tile_size_lvl0=tile_size_lvl0,
        tissue_frac_grid=tissue_frac_grid,
        artifact_frac_grid=artifact_frac_grid,
        tiles=tiles,
        rejected_tiles=rejected_tiles,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess WSI with tissue/artifact QC.")
    parser.add_argument("wsi", type=str, help="Path to WSI file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE_PX, help="Tile size (default 512)")
    parser.add_argument("--target-mag", type=float, default=TARGET_MAG, help="Target magnification (default 20x)")
    parser.add_argument("--min-tissue", type=float, default=MIN_TISSUE_FRAC, help="Minimum tissue fraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device for GrandQC (cuda/cpu)")

    args = parser.parse_args()

    wsi_path = Path(args.wsi)
    output_dir = Path(args.output) if args.output else None

    def progress(stage: str, current: int, total: int):
        if total > 0:
            pct = int(100 * current / total)
            print(f"\r[{stage}] {pct}%", end="", flush=True)
            if current == total:
                print()

    print(f"Processing: {wsi_path}")
    result = preprocess_wsi(
        wsi_path,
        output_dir=output_dir,
        tile_size_px=args.tile_size,
        target_mag=args.target_mag,
        min_tissue_frac=args.min_tissue,
        device=args.device,
        progress_callback=progress,
    )

    print(f"\nResults saved to: {result.output_dir}")
    print(f"  Grid shape: {result.grid_shape}")
    print(f"  Accepted tiles: {len(result.tiles)}")
    print(f"  Rejected tiles: {len(result.rejected_tiles)}")
    print(f"    - Low tissue: {result.metadata['rejected_low_tissue']}")
