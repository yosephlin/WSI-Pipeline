"""
QC-Gated Diversity Set Sampler (QG-DSS) for ROI selection.

This module implements a sampling strategy that selects ROIs which are:
  - High quality (tissue dense, low artifact)
  - Mutually diverse in feature space (cover different morphologies)

Algorithm:
  Step A: Generate candidate 16×16 tile windows within tissue groups
          until a target number of QC-passed candidates is reached
          (or a max-attempts cap is hit).
  Step B: Hard QC gate to reject low-tissue/high-artifact candidates
          (used when target_passed is None).
  Step C: Create ROI embeddings via weighted mean-pooling CONCH features
  Step D: Select a diverse subset via k-DPP

Usage:
    from pipeline.roi_sampler import sample_candidate_rois_selective

    result = sample_candidate_rois_selective(
        "slide_qc",
        target_passed=64,
    )
    passed = result.passed_candidates
    embeddings = result.embeddings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "ROICandidate",
    "ROISampler",
    "sample_candidate_rois_selective",
    "pick_roi_pair",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROI_SIZE = 16  # 16×16 tiles per ROI
DEFAULT_N_CANDIDATES = 128
DEFAULT_TARGET_PASSED = 64
DEFAULT_MIN_TISSUE_FRAC = 0.7
DEFAULT_TILE_MIN_TISSUE_FOR_VALID = 0.3
DEFAULT_MAX_ARTIFACT_FRAC = 0.05
DEFAULT_MAX_ARTIFACT_FRAC_FOR_GROUP = 0.25
DEFAULT_ARTIFACT_HARD_REJECT_FACTOR = 2.0
DEFAULT_MIN_VALID_FRAC = 0.5  # Require dense valid coverage per ROI window
DEFAULT_GROUP_COVER_MIN = 0.8  # Require ROI mostly inside tissue group
DEFAULT_TILE_MIN_TISSUE_FOR_GROUP = 0.2
DEFAULT_MERGE_RADIUS_TILES = 4
DEFAULT_MIN_GROUP_TILES_FRAC = 0.3
DEFAULT_K_SELECTED = 4  # Diverse ROI count per slide
DEFAULT_IOU_THRESHOLD = 0.3  # Post-selection NMS threshold in tile-grid coords
DEFAULT_IOU_PREV_MAX = 0.6  # Inter-epoch overlap block threshold
DEFAULT_NOVELTY_ALPHA = 1.0
DEFAULT_NOVELTY_EPSILON = 0.05


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ROICandidate:
    """A candidate ROI (16×16 tile window)."""
    roi_id: int
    grid_i: int  # Top-left row index in tile grid
    grid_j: int  # Top-left col index in tile grid
    # QC metrics (aggregated over the 16×16 window)
    tissue_frac: float
    artifact_frac: float
    group_id: Optional[int] = None
    # Embedding (populated in Step C)
    embedding: Optional[np.ndarray] = None
    # QC gate result
    passed_qc: bool = True
    reject_reason: Optional[str] = None


@dataclass
class ROISamplerResult:
    """Result of ROI sampling."""
    candidates: List[ROICandidate]
    passed_candidates: List[ROICandidate]
    rejected_candidates: List[ROICandidate]
    embeddings: Optional[np.ndarray] = None  # (N_passed, embed_dim)
    selected_candidates: Optional[List[ROICandidate]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def pick_roi_pair(result: ROISamplerResult) -> Optional[Tuple[ROICandidate, ROICandidate]]:
    """Pick a pair of ROIs from a sampler result with a simple fallback strategy."""
    selected = list(result.selected_candidates or [])
    passed = list(result.passed_candidates or [])
    if len(selected) >= 2:
        return selected[0], selected[1]
    if len(selected) == 1 and len(passed) >= 2:
        pool = [c for c in passed if c is not selected[0]]
        if pool:
            pool = sorted(pool, key=lambda c: float(c.tissue_frac) * float(1.0 - c.artifact_frac), reverse=True)
            roi_b = pool[0]
        else:
            roi_b = selected[0]
        return selected[0], roi_b
    if len(passed) >= 2:
        passed_sorted = sorted(
            passed,
            key=lambda c: float(c.tissue_frac) * float(1.0 - c.artifact_frac),
            reverse=True,
        )
        return passed_sorted[0], passed_sorted[1]
    if len(passed) == 1:
        return passed[0], passed[0]
    return None


# ---------------------------------------------------------------------------
# ROI Sampler
# ---------------------------------------------------------------------------


class ROISampler:
    """
    QC-Gated Diversity Set Sampler for ROI selection.

    Loads precomputed QC grids (from pipeline/preprocess_qc.py) and CONCH features,
    then generates diverse, high-quality ROI candidates.
    """

    def __init__(
        self,
        qc_zarr_path: Union[str, Path],
        features_path: Optional[Union[str, Path]] = None,
        roi_size: int = ROI_SIZE,
    ):
        """
        Initialize the ROI sampler.

        Args:
            qc_zarr_path: Path to qc_grids.zarr from pipeline/preprocess_qc.py
            features_path: Path to CONCH features Zarr store (.zarr) or None
            roi_size: Size of ROI window in tiles (default 16×16)
        """
        self.qc_zarr_path = Path(qc_zarr_path)
        self.features_path = Path(features_path) if features_path else None
        self.roi_size = roi_size

        self._group_info: Optional[List[Dict[str, Any]]] = None
        self._group_weights: Optional[np.ndarray] = None
        self._group_cache_key: Optional[Tuple[float, int, float, float, float]] = None

        # Load QC grids
        self._load_qc_grids()

        # Load features if provided
        self.features_zarr: Optional[Any] = None
        if self.features_path and self.features_path.exists():
            self._load_features()

        # QC masks and integral images (computed lazily per slide)
        self._integral_valid: Optional[np.ndarray] = None
        self._integral_tissue_mean: Optional[np.ndarray] = None
        self._integral_artifact_mean: Optional[np.ndarray] = None
        self._qc_cache_key: Optional[Tuple[float, float]] = None
        self._last_qc_thresholds: Optional[Dict[str, Optional[float]]] = None

    def _load_qc_grids(self) -> None:
        """Load QC grids from Zarr."""
        try:
            import zarr
        except ImportError as e:
            raise ImportError("zarr is required. Install via: pip install zarr") from e

        if not self.qc_zarr_path.exists():
            raise FileNotFoundError(f"QC Zarr not found: {self.qc_zarr_path}")

        try:
            root = zarr.open_group(str(self.qc_zarr_path), mode="r", zarr_format=3)
        except TypeError:
            root = zarr.open_group(str(self.qc_zarr_path), mode="r")

        self.tissue_frac_grid = np.asarray(root["tissue_frac"]).astype(np.float32)
        try:
            self.artifact_frac_grid = np.asarray(root["artifact_frac"]).astype(np.float32)
        except KeyError:
            self.artifact_frac_grid = None
        self.valid_mask_grid = None

        self.grid_shape = self.tissue_frac_grid.shape
        self.n_rows, self.n_cols = self.grid_shape

    def _load_features(self) -> None:
        """Load CONCH features from a Zarr store."""
        if self.features_path is None:
            return

        if not (self.features_path.suffix == ".zarr" or self.features_path.is_dir()):
            raise ValueError("features_path must point to a .zarr store.")

        import zarr

        root = zarr.open(str(self.features_path), mode="r")
        if hasattr(root, "shape"):
            arr = root
        else:
            arr = None
            for key in ["features", "embeddings", "arr_0"]:
                if key in root:
                    arr = root[key]
                    break
            if arr is None:
                keys = list(root.array_keys()) if hasattr(root, "array_keys") else list(root.keys())
                if not keys:
                    raise ValueError(f"No arrays found in features Zarr: {self.features_path}")
                arr = root[keys[0]]

        if len(getattr(arr, "shape", [])) != 3:
            raise ValueError("Features Zarr must be a 3D array (H, W, C).")

        self.features_zarr = arr

    def _prefix_sum_2d(self, arr: np.ndarray) -> np.ndarray:
        """Compute padded 2D prefix sum for fast window sums."""
        if arr.dtype.kind in ("b", "i", "u"):
            work = arr.astype(np.int64, copy=False)
            return np.pad(work, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        work = arr.astype(np.float64, copy=False)
        return np.pad(work, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    def _window_sum(self, integral: np.ndarray, i: int, j: int, size: int) -> float:
        """Compute the sum over a size x size window using a padded prefix sum."""
        i0, j0 = i, j
        i1, j1 = i + size, j + size
        return float(integral[i1, j1] - integral[i0, j1] - integral[i1, j0] + integral[i0, j0])

    def _ensure_qc_cache(
        self,
        tile_min_tissue_for_valid: float,
        max_artifact_frac: float,
    ) -> None:
        """Prepare QC masks and integral images for fast ROI window queries."""
        key = (
            float(tile_min_tissue_for_valid),
            float(max_artifact_frac),
        )
        if self._qc_cache_key == key:
            return

        if self.n_rows == 0 or self.n_cols == 0:
            empty = np.zeros((0, 0), dtype=bool)
            self.valid_mask_grid = empty
            self._integral_valid = np.zeros((1, 1), dtype=np.int32)
            self._integral_tissue_mean = np.zeros((1, 1), dtype=np.float32)
            self._integral_artifact_mean = np.zeros((1, 1), dtype=np.float32)
            self._qc_cache_key = key
            return

        tissue_mask = self.tissue_frac_grid >= float(tile_min_tissue_for_valid)
        if self.artifact_frac_grid is None:
            artifact_frac = np.zeros_like(self.tissue_frac_grid, dtype=np.float32)
        else:
            artifact_frac = self.artifact_frac_grid.astype(np.float32, copy=False)

        artifact_mask = artifact_frac > max_artifact_frac
        valid_mask = tissue_mask

        self.valid_mask_grid = valid_mask
        self._integral_valid = self._prefix_sum_2d(valid_mask.astype(np.uint8))
        self._integral_tissue_mean = self._prefix_sum_2d(self.tissue_frac_grid.astype(np.float32, copy=False))
        self._integral_artifact_mean = self._prefix_sum_2d(artifact_frac)
        self._qc_cache_key = key

    def _ensure_group_cache(
        self,
        *,
        tile_min_tissue_for_group: float = DEFAULT_TILE_MIN_TISSUE_FOR_GROUP,
        merge_radius_tiles: int = DEFAULT_MERGE_RADIUS_TILES,
        min_group_tiles_frac: float = DEFAULT_MIN_GROUP_TILES_FRAC,
        group_cover_min: float = DEFAULT_GROUP_COVER_MIN,
        max_artifact_frac_for_group: float = DEFAULT_MAX_ARTIFACT_FRAC_FOR_GROUP,
    ) -> None:
        """Prepare tissue groups directly on the tile grid."""
        key = (
            float(tile_min_tissue_for_group),
            int(merge_radius_tiles),
            float(min_group_tiles_frac),
            float(group_cover_min),
            float(max_artifact_frac_for_group),
        )
        if self._group_info is not None and self._group_cache_key == key:
            return

        try:
            import cv2
        except ImportError as e:
            raise ImportError("OpenCV (`cv2`) is required for tissue grouping.") from e

        if self.tissue_frac_grid.size == 0:
            self._group_info = []
            self._group_weights = np.zeros((0,), dtype=np.float32)
            self._group_cache_key = key
            return

        tile_min = float(tile_min_tissue_for_group)
        tile_min = min(max(tile_min, 0.0), 1.0)
        tile_tissue = self.tissue_frac_grid >= tile_min
        if self.artifact_frac_grid is not None:
            tile_tissue = tile_tissue & (self.artifact_frac_grid <= float(max_artifact_frac_for_group))

        merge_radius = int(merge_radius_tiles)
        if merge_radius > 0:
            k = 2 * merge_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            merged = cv2.dilate(tile_tissue.astype(np.uint8), kernel) > 0
        else:
            merged = tile_tissue

        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            merged.astype(np.uint8),
            connectivity=8,
        )
        if n_labels <= 1:
            self._group_info = []
            self._group_weights = np.zeros((0,), dtype=np.float32)
            self._group_cache_key = key
            return

        roi_size = int(self.roi_size)
        roi_area = float(roi_size * roi_size)
        cover_min = min(max(float(group_cover_min), 0.0), 1.0)
        min_group_frac = min(max(float(min_group_tiles_frac), 0.0), 1.0)
        min_group_tiles = roi_area * min_group_frac
        group_info: List[Dict[str, Any]] = []

        for gid in range(1, n_labels):
            x, y, w, h, area = stats[gid]
            if w <= 0 or h <= 0:
                continue
            group_area = int(area)
            if min_group_tiles > 0 and group_area < min_group_tiles:
                continue
            if h < roi_size or w < roi_size:
                continue

            min_i = int(y)
            min_j = int(x)
            max_i = min_i + int(h) - 1
            max_j = min_j + int(w) - 1

            mask_crop = labels[min_i:max_i + 1, min_j:max_j + 1] == gid
            if not mask_crop.any():
                continue

            prefix = self._prefix_sum_2d(mask_crop.astype(np.uint8))
            window_sum = (
                prefix[roi_size:, roi_size:]
                - prefix[:-roi_size, roi_size:]
                - prefix[roi_size:, :-roi_size]
                + prefix[:-roi_size, :-roi_size]
            )
            valid_windows = window_sum >= roi_area * cover_min
            n_placements = int(np.count_nonzero(valid_windows))
            if n_placements == 0:
                continue

            group_info.append(
                {
                    "group_id": int(gid),
                    "bbox": (int(min_i), int(max_i), int(min_j), int(max_j)),
                    "prefix": prefix,
                    "origin": (int(min_i), int(min_j)),
                    "shape": (int(h), int(w)),
                    "weight": float(np.sqrt(n_placements)),
                    "area": group_area,
                    "n_placements": n_placements,
                }
            )

        self._group_info = group_info
        if group_info:
            self._group_weights = np.array([g["weight"] for g in group_info], dtype=np.float32)
        else:
            self._group_weights = np.zeros((0,), dtype=np.float32)
        self._group_cache_key = key

    def _compute_roi_qc_metrics(
        self,
        i: int,
        j: int,
    ) -> Tuple[float, float]:
        """
        Compute QC metrics for an ROI window.

        Args:
            i, j: Top-left position in tile grid

        Returns:
            (tissue_mean, artifact_mean)
        """
        roi_size = self.roi_size
        if self._integral_valid is None:
            raise ValueError("QC cache not initialized. Call _ensure_qc_cache first.")

        roi_area = float(roi_size * roi_size)
        if self._integral_tissue_mean is None or self._integral_artifact_mean is None:
            raise ValueError("QC cache not initialized. Call _ensure_qc_cache first.")

        tissue_sum = self._window_sum(self._integral_tissue_mean, i, j, roi_size)
        artifact_sum = self._window_sum(self._integral_artifact_mean, i, j, roi_size)

        tissue_mean = tissue_sum / roi_area if roi_area > 0 else 0.0
        artifact_mean = artifact_sum / roi_area if roi_area > 0 else 0.0

        return tissue_mean, artifact_mean

    def _normalize_percentile(self, value: Optional[float]) -> Optional[float]:
        """Normalize percentile inputs; return None if disabled."""
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            return None
        if value > 100:
            raise ValueError("Percentile must be in (0, 100].")
        return value

    def _resolve_qc_thresholds(
        self,
        tissue_values: np.ndarray,
        artifact_values: np.ndarray,
        min_tissue_frac: float,
        max_artifact_frac: float,
        tissue_percentile: Optional[float],
        artifact_percentile: Optional[float],
    ) -> Dict[str, Optional[float]]:
        """Compute absolute/slide-relative QC thresholds."""
        tissue_pct = self._normalize_percentile(tissue_percentile)
        artifact_pct = self._normalize_percentile(artifact_percentile)

        tissue_thr = float(min_tissue_frac)
        artifact_thr = float(max_artifact_frac)

        if tissue_pct is not None and tissue_values.size > 0:
            tissue_thr = float(np.nanpercentile(tissue_values, tissue_pct))
        if artifact_pct is not None and artifact_values.size > 0:
            artifact_thr = float(np.nanpercentile(artifact_values, artifact_pct))
            artifact_thr = min(artifact_thr, float(max_artifact_frac))

        if not np.isfinite(tissue_thr):
            tissue_thr = float(min_tissue_frac)
        if not np.isfinite(artifact_thr):
            artifact_thr = float(max_artifact_frac)

        tissue_thr = min(max(tissue_thr, 0.0), 1.0)
        artifact_thr = min(max(artifact_thr, 0.0), 1.0)

        return {
            "tissue_threshold": tissue_thr,
            "artifact_threshold": artifact_thr,
            "tissue_percentile": tissue_pct,
            "artifact_percentile": artifact_pct,
        }

    def _apply_qc_thresholds(
        self,
        candidate: ROICandidate,
        tissue_threshold: float,
        artifact_threshold: float,
    ) -> bool:
        """Apply QC thresholds to a candidate and update its state."""
        candidate.passed_qc = True
        candidate.reject_reason = None
        if candidate.tissue_frac < tissue_threshold:
            candidate.passed_qc = False
            candidate.reject_reason = "low_tissue"
        elif candidate.artifact_frac > artifact_threshold:
            candidate.passed_qc = False
            candidate.reject_reason = "high_artifact"
        return candidate.passed_qc

    def generate_candidates(
        self,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        target_passed: Optional[int] = None,
        max_attempts: Optional[int] = None,
        seed: Optional[int] = None,
        min_tissue_frac: float = DEFAULT_MIN_TISSUE_FRAC,
        max_artifact_frac: float = DEFAULT_MAX_ARTIFACT_FRAC,
        tile_min_tissue_for_valid: float = DEFAULT_TILE_MIN_TISSUE_FOR_VALID,
        min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
        group_cover_min: float = DEFAULT_GROUP_COVER_MIN,
        tile_min_tissue_for_group: float = DEFAULT_TILE_MIN_TISSUE_FOR_GROUP,
        merge_radius_tiles: int = DEFAULT_MERGE_RADIUS_TILES,
        min_group_tiles_frac: float = DEFAULT_MIN_GROUP_TILES_FRAC,
        max_artifact_frac_for_group: float = DEFAULT_MAX_ARTIFACT_FRAC_FOR_GROUP,
        tissue_percentile: Optional[float] = None,
        artifact_percentile: Optional[float] = None,
    ) -> List[ROICandidate]:
        """
        Step A: Generate M candidate 16×16 ROI windows within tissue groups.
        Uses rejection sampling over group bounds to avoid enumerating all positions.

        Args:
            n_candidates: Number of candidate ROIs to sample (used when target_passed is None)
            target_passed: Target number of QC-passed candidates to collect
            max_attempts: Max sampling attempts before stopping
            seed: Random seed for reproducibility
            min_tissue_frac: Minimum ROI mean tissue fraction threshold (applied in Step B)
            max_artifact_frac: Maximum ROI mean artifact fraction threshold
            tile_min_tissue_for_valid: Minimum tile tissue fraction to count as valid
            min_valid_frac: Minimum valid tile coverage fraction for ROI window
            group_cover_min: Minimum fraction of ROI tiles inside a tissue group
            tile_min_tissue_for_group: Minimum tile tissue fraction to build groups
            merge_radius_tiles: Dilation radius (tiles) for merging nearby tissue
            min_group_tiles_frac: Minimum group size as fraction of ROI area
            max_artifact_frac_for_group: Maximum artifact fraction for group formation
            tissue_percentile: Slide-relative tissue threshold percentile (lower tail)
            artifact_percentile: Slide-relative artifact threshold percentile (upper tail)
        Returns:
            List of ROICandidate objects with QC metrics computed
        """
        rng = np.random.default_rng(seed)

        self._ensure_qc_cache(
            tile_min_tissue_for_valid=tile_min_tissue_for_valid,
            max_artifact_frac=max_artifact_frac,
        )
        self._ensure_group_cache(
            tile_min_tissue_for_group=tile_min_tissue_for_group,
            merge_radius_tiles=merge_radius_tiles,
            min_group_tiles_frac=min_group_tiles_frac,
            group_cover_min=group_cover_min,
            max_artifact_frac_for_group=max_artifact_frac_for_group,
        )

        if not self._group_info:
            return []

        candidates: List[ROICandidate] = []
        seen: set[Tuple[int, int]] = set()
        roi_size = self.roi_size
        roi_area = float(roi_size * roi_size)
        if target_passed is not None:
            target_passed = max(0, int(target_passed))
        if max_attempts is None:
            if target_passed is not None:
                max_attempts = max(int(target_passed) * 200, 1000)
            else:
                max_attempts = max(int(n_candidates) * 50, 1000)
        attempts = 0
        passed_count = 0

        weights = self._group_weights
        if weights is None or weights.size == 0 or float(weights.sum()) <= 0:
            return []
        probs = weights / float(weights.sum())

        tissue_pct = self._normalize_percentile(tissue_percentile)
        artifact_pct = self._normalize_percentile(artifact_percentile)
        slide_relative = tissue_pct is not None or artifact_pct is not None
        hard_reject_thr = None
        if target_passed is not None:
            max_artifact = float(max_artifact_frac)
            if max_artifact > 0:
                hard_reject_thr = max_artifact * DEFAULT_ARTIFACT_HARD_REJECT_FACTOR

        def _sample_candidate() -> Optional[ROICandidate]:
            nonlocal attempts
            attempts += 1
            group_idx = int(rng.choice(len(self._group_info), p=probs))
            info = self._group_info[group_idx]
            min_i, max_i, min_j, max_j = info["bbox"]
            max_i0 = max_i - roi_size + 1
            max_j0 = max_j - roi_size + 1
            if max_i0 < min_i or max_j0 < min_j:
                return None
            i = int(rng.integers(min_i, max_i0 + 1))
            j = int(rng.integers(min_j, max_j0 + 1))
            if (i, j) in seen:
                return None

            origin_i, origin_j = info.get("origin", (0, 0))
            local_i = i - int(origin_i)
            local_j = j - int(origin_j)
            shape = info.get("shape")
            if shape is not None:
                if local_i < 0 or local_j < 0:
                    return None
                if (local_i + roi_size) > int(shape[0]) or (local_j + roi_size) > int(shape[1]):
                    return None
            if self._window_sum(info["prefix"], local_i, local_j, roi_size) < roi_area * float(group_cover_min):
                return None

            if self._integral_valid is None:
                raise ValueError("QC cache not initialized. Call _ensure_qc_cache first.")
            valid_count = self._window_sum(self._integral_valid, i, j, roi_size)
            if valid_count / roi_area < float(min_valid_frac):
                return None

            tissue_frac, artifact_frac = self._compute_roi_qc_metrics(i, j)
            if hard_reject_thr is not None and artifact_frac > hard_reject_thr:
                return None
            candidate = ROICandidate(
                roi_id=len(candidates),
                grid_i=i,
                grid_j=j,
                tissue_frac=tissue_frac,
                artifact_frac=artifact_frac,
                group_id=int(info["group_id"]) if "group_id" in info else None,
            )
            seen.add((i, j))
            return candidate

        if target_passed is None:
            while len(candidates) < int(n_candidates) and attempts < max_attempts:
                candidate = _sample_candidate()
                if candidate is None:
                    continue
                candidates.append(candidate)
            self._last_qc_thresholds = None
            return candidates

        thresholds = self._resolve_qc_thresholds(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            min_tissue_frac,
            max_artifact_frac,
            tissue_pct,
            artifact_pct,
        )

        if slide_relative:
            calib_target = max(32, min(int(target_passed) * 2, 512))
            if max_attempts is not None:
                calib_target = min(calib_target, int(max_attempts))
            while len(candidates) < calib_target and attempts < max_attempts:
                candidate = _sample_candidate()
                if candidate is None:
                    continue
                candidates.append(candidate)
            if candidates:
                tissue_vals = np.array([c.tissue_frac for c in candidates], dtype=np.float32)
                artifact_vals = np.array([c.artifact_frac for c in candidates], dtype=np.float32)
                thresholds = self._resolve_qc_thresholds(
                    tissue_vals,
                    artifact_vals,
                    min_tissue_frac,
                    max_artifact_frac,
                    tissue_pct,
                    artifact_pct,
                )

        tissue_thr = float(thresholds["tissue_threshold"])
        artifact_thr = float(thresholds["artifact_threshold"])
        self._last_qc_thresholds = thresholds

        for candidate in candidates:
            if self._apply_qc_thresholds(candidate, tissue_thr, artifact_thr):
                passed_count += 1

        while passed_count < int(target_passed) and attempts < max_attempts:
            candidate = _sample_candidate()
            if candidate is None:
                continue
            if self._apply_qc_thresholds(candidate, tissue_thr, artifact_thr):
                passed_count += 1
            candidates.append(candidate)

        return candidates

    def apply_qc_gate(
        self,
        candidates: List[ROICandidate],
        min_tissue_frac: float = DEFAULT_MIN_TISSUE_FRAC,
        max_artifact_frac: float = DEFAULT_MAX_ARTIFACT_FRAC,
        tile_min_tissue_for_valid: float = DEFAULT_TILE_MIN_TISSUE_FOR_VALID,
        tissue_percentile: Optional[float] = None,
        artifact_percentile: Optional[float] = None,
    ) -> Tuple[List[ROICandidate], List[ROICandidate]]:
        """
        Step B: Apply hard QC gate to reject worst candidates.

        Keeps ROI if:
          - tissue_mean >= min_tissue_frac (default 0.7)
          - artifact_mean <= max_artifact_frac (default 0.05)

        Args:
            candidates: List of ROICandidate from generate_candidates()
            min_tissue_frac: Minimum ROI mean tissue fraction threshold
            max_artifact_frac: Maximum ROI mean artifact fraction threshold
            tile_min_tissue_for_valid: Minimum tile tissue fraction to count as valid
            tissue_percentile: Slide-relative tissue threshold percentile (lower tail)
            artifact_percentile: Slide-relative artifact threshold percentile (upper tail)

        Returns:
            Tuple of (passed_candidates, rejected_candidates)
        """
        if len(candidates) == 0:
            self._last_qc_thresholds = None
            return [], []

        self._ensure_qc_cache(
            tile_min_tissue_for_valid=tile_min_tissue_for_valid,
            max_artifact_frac=max_artifact_frac,
        )

        tissue_vals = []
        artifact_vals = []
        for candidate in candidates:
            tissue_frac, artifact_frac = self._compute_roi_qc_metrics(
                candidate.grid_i,
                candidate.grid_j,
            )
            candidate.tissue_frac = tissue_frac
            candidate.artifact_frac = artifact_frac
            tissue_vals.append(tissue_frac)
            artifact_vals.append(artifact_frac)

        thresholds = self._resolve_qc_thresholds(
            np.array(tissue_vals, dtype=np.float32),
            np.array(artifact_vals, dtype=np.float32),
            min_tissue_frac,
            max_artifact_frac,
            tissue_percentile,
            artifact_percentile,
        )
        self._last_qc_thresholds = thresholds
        tissue_thr = float(thresholds["tissue_threshold"])
        artifact_thr = float(thresholds["artifact_threshold"])

        passed = []
        rejected = []
        for candidate in candidates:
            if self._apply_qc_thresholds(candidate, tissue_thr, artifact_thr):
                passed.append(candidate)
            else:
                rejected.append(candidate)

        return passed, rejected

    def compute_roi_embeddings(
        self,
        candidates: List[ROICandidate],
    ) -> np.ndarray:
        """
        Step C: Create ROI embeddings via mean-pooling CONCH features.

        For each ROI:
          e_i = weighted_mean(CONCH_features[ROI])

        Args:
            candidates: List of ROICandidate that passed QC

        Returns:
            Embeddings array of shape (N_candidates, embed_dim)
        """
        if self.features_zarr is None:
            raise ValueError(
            "Features not loaded. Provide a .zarr features_path."
            )

        if len(candidates) == 0:
            return np.array([])

        if self._integral_valid is None:
            self._ensure_qc_cache(
                tile_min_tissue_for_valid=DEFAULT_TILE_MIN_TISSUE_FOR_VALID,
                max_artifact_frac=DEFAULT_MAX_ARTIFACT_FRAC,
            )

        roi_size = self.roi_size
        feat_dim = int(self.features_zarr.shape[2])

        def _get_roi_features(i: int, j: int) -> np.ndarray:
            return np.asarray(
                self.features_zarr[i:i + roi_size, j:j + roi_size, :],
                dtype=np.float32,
            )

        embeddings = []

        for candidate in candidates:
            i, j = candidate.grid_i, candidate.grid_j

            # Extract ROI features
            roi_features = _get_roi_features(i, j)  # (roi, roi, feat_dim)
            roi_valid = self.valid_mask_grid[i:i + roi_size, j:j + roi_size]  # (16, 16)

            # Weighted mean pool over valid tiles
            if roi_valid.any():
                tissue_roi = self.tissue_frac_grid[i:i + roi_size, j:j + roi_size]
                if self.artifact_frac_grid is None:
                    artifact_roi = np.zeros_like(tissue_roi, dtype=np.float32)
                else:
                    artifact_roi = self.artifact_frac_grid[i:i + roi_size, j:j + roi_size]
                weights = tissue_roi * (1.0 - artifact_roi)
                weights = np.clip(weights, 0.0, 1.0)

                valid_features = roi_features[roi_valid]  # (N_valid, feat_dim)
                valid_weights = weights[roi_valid].astype(np.float32)
                weight_sum = float(valid_weights.sum())
                if weight_sum > 0:
                    embedding = (valid_features * valid_weights[:, None]).sum(axis=0) / weight_sum
                else:
                    embedding = valid_features.mean(axis=0)  # (feat_dim,)
            else:
                embedding = np.zeros(feat_dim, dtype=np.float32)

            embeddings.append(embedding)
            candidate.embedding = embedding

        embeddings = np.stack(embeddings, axis=0)  # (N_candidates, feat_dim)

        return embeddings

    def _compute_quality_scores(self, candidates: List[ROICandidate]) -> np.ndarray:
        """Compute quality scores q_i for k-DPP selection."""
        tissue = np.array([c.tissue_frac for c in candidates], dtype=np.float32)
        artifact = np.array([c.artifact_frac for c in candidates], dtype=np.float32)
        if tissue.size == 0:
            return np.array([], dtype=np.float32)

        # q_i = tissue_frac * (1 - artifact_frac)
        quality = tissue * (1.0 - artifact)
        return np.clip(quality, 0.0, None).astype(np.float32)

    def _compute_novelty_weights(
        self,
        embeddings: np.ndarray,
        recent_embeds: Optional[np.ndarray],
        alpha: float,
        epsilon: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute novelty weights from recent embeddings."""
        if recent_embeds is None:
            return np.ones((embeddings.shape[0],), dtype=np.float32), None

        recent = np.asarray(recent_embeds, dtype=np.float32)
        if recent.ndim != 2:
            raise ValueError("recent_embeds must be a 2D array.")
        if recent.shape[0] == 0:
            return np.ones((embeddings.shape[0],), dtype=np.float32), None
        if recent.shape[1] != embeddings.shape[1]:
            raise ValueError("recent_embeds must match embedding dimension.")

        emb = embeddings.astype(np.float32, copy=False)
        emb_norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_norm = np.where(emb_norm == 0, 1.0, emb_norm)
        emb_unit = emb / emb_norm

        rec_norm = np.linalg.norm(recent, axis=1, keepdims=True)
        rec_norm = np.where(rec_norm == 0, 1.0, rec_norm)
        rec_unit = recent / rec_norm

        sims = emb_unit @ rec_unit.T
        max_sim = np.max(sims, axis=1)
        max_sim = np.clip(max_sim, -1.0, 1.0)
        novelty = 1.0 - max_sim
        novelty = np.clip(novelty, 0.0, 1.0)

        alpha = float(alpha)
        epsilon = float(epsilon)
        if alpha <= 0:
            alpha = 1.0
        if epsilon < 0:
            epsilon = 0.0

        weights = (epsilon + novelty) ** alpha
        return weights.astype(np.float32), max_sim.astype(np.float32)

    def _build_dpp_kernel(
        self,
        embeddings: np.ndarray,
        quality: np.ndarray,
        sigma: Optional[float],
        positions: Optional[np.ndarray] = None,
        tau: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, Optional[float]]:
        """Build L-ensemble kernel for k-DPP selection."""
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array for k-DPP.")
        if embeddings.shape[0] != quality.shape[0]:
            raise ValueError("Embeddings and quality scores must align.")

        emb = embeddings.astype(np.float64, copy=False)
        sq_norms = np.sum(emb * emb, axis=1, keepdims=True)
        dists = sq_norms + sq_norms.T - 2.0 * (emb @ emb.T)
        np.maximum(dists, 0.0, out=dists)

        if sigma is None:
            if dists.shape[0] <= 1:
                sigma = 1.0
            else:
                upper = dists[np.triu_indices(dists.shape[0], k=1)]
                nonzero = upper[upper > 0]
                if nonzero.size == 0:
                    sigma = 1.0
                else:
                    sigma = float(np.sqrt(np.median(nonzero)))
                    if sigma <= 0:
                        sigma = 1.0

        if sigma <= 0:
            raise ValueError("sigma must be > 0 for k-DPP.")

        # L_ij = (q_i q_j) * exp(-||e_i - e_j||^2 / sigma^2)
        kernel = np.exp(-dists / (sigma * sigma))
        tau_used: Optional[float] = None
        if positions is not None:
            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError("positions must be an (N, 2) array of ROI centers.")
            if tau is None:
                tau = float(self.roi_size) * 1.5
            tau_used = float(tau)
            if tau_used <= 0:
                raise ValueError("tau must be > 0 for spatial repulsion.")
            pos = positions.astype(np.float64, copy=False)
            diffs = pos[:, None, :] - pos[None, :, :]
            d2 = np.sum(diffs * diffs, axis=-1)
            spatial = np.exp(-d2 / (tau_used * tau_used))
            kernel = kernel * spatial
        q = quality.astype(np.float64, copy=False)
        L = (q[:, None] * q[None, :]) * kernel
        return L, float(sigma), tau_used

    def _quality_fps(
        self,
        embeddings: np.ndarray,
        quality: np.ndarray,
        k: int,
        seed: Optional[int],
    ) -> List[int]:
        """Quality-weighted farthest-point sampling fallback."""
        n = int(embeddings.shape[0])
        if n == 0 or k <= 0:
            return []
        k = min(int(k), n)
        rng = np.random.default_rng(seed)

        q = np.clip(np.asarray(quality, dtype=np.float32), 0.0, None)
        if np.all(q <= 0):
            start = int(rng.integers(0, n))
        else:
            max_q = float(q.max())
            candidates = np.where(q == max_q)[0]
            start = int(rng.choice(candidates))

        selected = [start]
        dists = np.sum((embeddings - embeddings[start]) ** 2, axis=1)

        while len(selected) < k:
            score = dists if np.all(q <= 0) else dists * q
            score = score.astype(np.float64, copy=False)
            score[selected] = -1.0
            next_idx = int(np.argmax(score))
            if next_idx in selected:
                break
            selected.append(next_idx)
            new_dists = np.sum((embeddings - embeddings[next_idx]) ** 2, axis=1)
            dists = np.minimum(dists, new_dists)

        return selected

    def _roi_iou(self, a: ROICandidate, b: ROICandidate) -> float:
        """Compute IoU between two ROI boxes in tile-grid coordinates."""
        size = int(self.roi_size)
        ax0, ay0 = int(a.grid_j), int(a.grid_i)
        bx0, by0 = int(b.grid_j), int(b.grid_i)
        ax1, ay1 = ax0 + size, ay0 + size
        bx1, by1 = bx0 + size, by0 + size

        inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
        inter_h = max(0, min(ay1, by1) - max(ay0, by0))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area = size * size
        union = (2 * area) - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def _roi_iou_coords(self, candidate: ROICandidate, i: int, j: int) -> float:
        """Compute IoU between a candidate and a grid (i, j) top-left."""
        size = int(self.roi_size)
        ax0, ay0 = int(candidate.grid_j), int(candidate.grid_i)
        bx0, by0 = int(j), int(i)
        ax1, ay1 = ax0 + size, ay0 + size
        bx1, by1 = bx0 + size, by0 + size

        inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
        inter_h = max(0, min(ay1, by1) - max(ay0, by0))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area = size * size
        union = (2 * area) - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def _apply_iou_nms(
        self,
        ordered: List[ROICandidate],
        iou_threshold: float,
    ) -> List[ROICandidate]:
        """Filter ordered candidates with a simple IoU-based NMS."""
        if iou_threshold <= 0:
            return list(ordered)
        kept: List[ROICandidate] = []
        for candidate in ordered:
            if all(self._roi_iou(candidate, prev) <= iou_threshold for prev in kept):
                kept.append(candidate)
        return kept

    def select_diverse_rois(
        self,
        candidates: List[ROICandidate],
        k: int,
        sigma: Optional[float] = None,
        tau: Optional[float] = None,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        recent_rois: Optional[List[Tuple[int, int]]] = None,
        recent_embeds: Optional[np.ndarray] = None,
        recent_group_ids: Optional[List[int]] = None,
        novelty_alpha: float = DEFAULT_NOVELTY_ALPHA,
        novelty_epsilon: float = DEFAULT_NOVELTY_EPSILON,
        iou_prev_max: float = DEFAULT_IOU_PREV_MAX,
        enforce_group_spread: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[List[ROICandidate], Dict[str, Any]]:
        """
        Step D: Select a diverse, high-quality subset using k-DPP.

        Args:
            candidates: List of ROICandidate with embeddings populated
            k: Number of ROIs to select
            sigma: Kernel bandwidth (auto if None)
            tau: Spatial kernel bandwidth in tile units (default roi_size * 1.5)
            iou_threshold: IoU threshold for post-selection NMS
            recent_rois: Recent ROI top-left coords (grid_i, grid_j) from last epoch
            recent_embeds: Recent ROI embeddings for novelty penalty
            recent_group_ids: Recent ROI group ids (optional)
            novelty_alpha: Exponent for novelty penalty
            novelty_epsilon: Epsilon added to novelty (prevents zero quality)
            iou_prev_max: IoU threshold for blocking overlap with previous epoch ROIs
            enforce_group_spread: Attempt to spread selections across groups
            seed: Random seed for k-DPP sampling

        Returns:
            Tuple of (selected_candidates, selection_metadata)
        """
        if k <= 0 or len(candidates) == 0:
            return [], {"selected_indices": [], "selection_k": k, "selection_sigma": None, "selection_tau": None}

        if iou_prev_max is None:
            iou_prev_max = 0.0

        embeddings = []
        for candidate in candidates:
            if candidate.embedding is None:
                raise ValueError("Embeddings are required for k-DPP selection.")
            embeddings.append(candidate.embedding)
        embeddings = np.stack(embeddings, axis=0)

        base_quality = self._compute_quality_scores(candidates)
        novelty_weights, novelty_sim = self._compute_novelty_weights(
            embeddings,
            recent_embeds,
            novelty_alpha,
            novelty_epsilon,
        )
        quality = base_quality * novelty_weights
        id_to_index = {id(c): i for i, c in enumerate(candidates)}
        positions = np.array(
            [
                (c.grid_i + (self.roi_size / 2.0), c.grid_j + (self.roi_size / 2.0))
                for c in candidates
            ],
            dtype=np.float64,
        )
        L, sigma_used, tau_used = self._build_dpp_kernel(
            embeddings,
            quality,
            sigma,
            positions=positions,
            tau=tau,
        )
        L = 0.5 * (L + L.T)
        L += 1e-6 * np.eye(L.shape[0])

        selection_method = "k-dpp"
        fallback_reason = None
        if len(candidates) <= k:
            selection_method = "all"
            indices = list(range(len(candidates)))
        else:
            try:
                if not np.all(np.isfinite(L)):
                    raise ValueError("Non-finite values in DPP kernel.")
                from dppy.finite_dpps import FiniteDPP

                dpp = FiniteDPP("likelihood", L=L)
                indices = list(dpp.sample_exact_k_dpp(size=k, random_state=seed))
            except Exception as exc:
                selection_method = "fps"
                fallback_reason = str(exc)
                indices = self._quality_fps(embeddings, quality, k, seed)

            if len(indices) < k:
                remaining = [i for i in np.argsort(-quality) if i not in indices]
                indices.extend(remaining[: max(0, k - len(indices))])

        selected = [candidates[i] for i in indices]
        selected = self._apply_iou_nms(selected, float(iou_threshold))

        recent_list: List[Tuple[int, int]] = []
        if recent_rois:
            for pair in recent_rois:
                if pair is None:
                    continue
                try:
                    recent_list.append((int(pair[0]), int(pair[1])))
                except Exception:
                    continue

        def _passes_prev(candidate: ROICandidate) -> bool:
            if not recent_list or iou_prev_max <= 0:
                return True
            return all(self._roi_iou_coords(candidate, i, j) <= float(iou_prev_max) for i, j in recent_list)

        if recent_list and iou_prev_max > 0:
            selected = [c for c in selected if _passes_prev(c)]

        selected_ids = {id(c) for c in selected}
        remaining = [i for i in np.argsort(-quality) if id(candidates[i]) not in selected_ids]
        if len(selected) < k:
            for idx in remaining:
                cand = candidates[idx]
                if not _passes_prev(cand):
                    continue
                if all(self._roi_iou(cand, prev) <= float(iou_threshold) for prev in selected):
                    selected.append(cand)
                if len(selected) >= k:
                    break

        group_swap = False
        if enforce_group_spread and len(selected) >= 2:
            selected_groups = [c.group_id for c in selected]
            if all(g is not None for g in selected_groups):
                if len(set(selected_groups)) == 1:
                    scores = [float(quality[id_to_index[id(c)]]) for c in selected]
                    drop_idx = int(np.argmin(scores))
                    base_selected = [c for j, c in enumerate(selected) if j != drop_idx]
                    base_groups = {c.group_id for c in base_selected}
                    for idx in remaining:
                        cand = candidates[idx]
                        if cand.group_id is None or cand.group_id in base_groups:
                            continue
                        if not _passes_prev(cand):
                            continue
                        if all(self._roi_iou(cand, prev) <= float(iou_threshold) for prev in base_selected):
                            selected = base_selected + [cand]
                            group_swap = True
                            break

        final_indices = [id_to_index[id(c)] for c in selected]

        selection_meta = {
            "selected_indices": final_indices,
            "selection_k": k,
            "selection_sigma": sigma_used,
            "selection_tau": tau_used,
            "selection_iou_threshold": float(iou_threshold),
            "selection_iou_prev_max": float(iou_prev_max),
            "selection_iou_kept": len(selected),
            "selection_method": selection_method,
            "fallback_reason": fallback_reason,
            "quality_score_min": float(quality.min()) if quality.size > 0 else None,
            "quality_score_max": float(quality.max()) if quality.size > 0 else None,
            "base_quality_min": float(base_quality.min()) if base_quality.size > 0 else None,
            "base_quality_max": float(base_quality.max()) if base_quality.size > 0 else None,
            "novelty_alpha": float(novelty_alpha),
            "novelty_epsilon": float(novelty_epsilon),
            "novelty_weight_min": float(novelty_weights.min()) if novelty_weights.size > 0 else None,
            "novelty_weight_max": float(novelty_weights.max()) if novelty_weights.size > 0 else None,
            "novelty_sim_min": float(np.min(novelty_sim)) if novelty_sim is not None else None,
            "novelty_sim_max": float(np.max(novelty_sim)) if novelty_sim is not None else None,
            "recent_count": len(recent_list),
            "recent_group_count": len(recent_group_ids) if recent_group_ids else 0,
            "group_spread_enforced": bool(enforce_group_spread),
            "group_spread_swap": bool(group_swap),
        }
        return selected, selection_meta

    def sample(
        self,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        target_passed: Optional[int] = DEFAULT_TARGET_PASSED,
        max_attempts: Optional[int] = None,
        min_tissue_frac: float = DEFAULT_MIN_TISSUE_FRAC,
        max_artifact_frac: float = DEFAULT_MAX_ARTIFACT_FRAC,
        tile_min_tissue_for_valid: float = DEFAULT_TILE_MIN_TISSUE_FOR_VALID,
        min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
        group_cover_min: float = DEFAULT_GROUP_COVER_MIN,
        tile_min_tissue_for_group: float = DEFAULT_TILE_MIN_TISSUE_FOR_GROUP,
        merge_radius_tiles: int = DEFAULT_MERGE_RADIUS_TILES,
        min_group_tiles_frac: float = DEFAULT_MIN_GROUP_TILES_FRAC,
        max_artifact_frac_for_group: float = DEFAULT_MAX_ARTIFACT_FRAC_FOR_GROUP,
        tissue_percentile: Optional[float] = None,
        artifact_percentile: Optional[float] = None,
        k: Optional[int] = DEFAULT_K_SELECTED,
        sigma: Optional[float] = None,
        tau: Optional[float] = None,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        recent_rois: Optional[List[Tuple[int, int]]] = None,
        recent_embeds: Optional[np.ndarray] = None,
        recent_group_ids: Optional[List[int]] = None,
        novelty_alpha: float = DEFAULT_NOVELTY_ALPHA,
        novelty_epsilon: float = DEFAULT_NOVELTY_EPSILON,
        iou_prev_max: float = DEFAULT_IOU_PREV_MAX,
        enforce_group_spread: bool = False,
        seed: Optional[int] = None,
    ) -> ROISamplerResult:
        """
        Run the full sampling pipeline (Steps A, B, C, D).

        Args:
            n_candidates: Number of candidate ROIs to sample (used when target_passed is None)
            target_passed: Target number of QC-passed candidates to collect
            max_attempts: Max sampling attempts before stopping
            min_tissue_frac: Minimum ROI mean tissue fraction for QC gate
            max_artifact_frac: Maximum ROI mean artifact fraction for QC gate
            tile_min_tissue_for_valid: Minimum tile tissue fraction to count as valid
            min_valid_frac: Minimum valid tile coverage fraction for ROI window
            group_cover_min: Minimum fraction of ROI tiles inside a tissue group
            tile_min_tissue_for_group: Minimum tile tissue fraction to build groups
            merge_radius_tiles: Dilation radius (tiles) for merging nearby tissue
            min_group_tiles_frac: Minimum group size as fraction of ROI area
            max_artifact_frac_for_group: Maximum artifact fraction for group formation
            tissue_percentile: Slide-relative tissue threshold percentile (lower tail)
            artifact_percentile: Slide-relative artifact threshold percentile (upper tail)
            k: Number of diverse ROIs to select (Step D)
            sigma: Kernel bandwidth for k-DPP (auto if None)
            tau: Spatial kernel bandwidth for k-DPP (default roi_size * 1.5)
            iou_threshold: IoU threshold for post-selection NMS
            recent_rois: Recent ROI top-left coords (grid_i, grid_j) from last epoch
            recent_embeds: Recent ROI embeddings for novelty penalty
            recent_group_ids: Recent ROI group ids (optional)
            novelty_alpha: Exponent for novelty penalty
            novelty_epsilon: Epsilon added to novelty (prevents zero quality)
            iou_prev_max: IoU threshold for blocking overlap with previous epoch ROIs
            enforce_group_spread: Attempt to spread selections across groups
            seed: Random seed

        Returns:
            ROISamplerResult with candidates, embeddings, selection, and metadata
        """
        if target_passed is not None and int(target_passed) <= 0:
            target_passed = None

        # Step A: Generate candidates
        candidates = self.generate_candidates(
            n_candidates=n_candidates,
            target_passed=target_passed,
            max_attempts=max_attempts,
            min_tissue_frac=min_tissue_frac,
            max_artifact_frac=max_artifact_frac,
            tile_min_tissue_for_valid=tile_min_tissue_for_valid,
            min_valid_frac=min_valid_frac,
            group_cover_min=group_cover_min,
            tile_min_tissue_for_group=tile_min_tissue_for_group,
            merge_radius_tiles=merge_radius_tiles,
            min_group_tiles_frac=min_group_tiles_frac,
            max_artifact_frac_for_group=max_artifact_frac_for_group,
            tissue_percentile=tissue_percentile,
            artifact_percentile=artifact_percentile,
            seed=seed,
        )

        # Step B: Apply QC gate (or reuse precomputed flags when target_passed is set)
        if target_passed is None:
            passed, rejected = self.apply_qc_gate(
                candidates,
                min_tissue_frac=min_tissue_frac,
                max_artifact_frac=max_artifact_frac,
                tile_min_tissue_for_valid=tile_min_tissue_for_valid,
                tissue_percentile=tissue_percentile,
                artifact_percentile=artifact_percentile,
            )
        else:
            passed = [c for c in candidates if c.passed_qc]
            rejected = [c for c in candidates if not c.passed_qc]

        # Step C: Compute embeddings (if features available)
        embeddings = None
        if self.features_zarr is not None and len(passed) > 0:
            embeddings = self.compute_roi_embeddings(passed)

        # Step D: Diverse subset selection via k-DPP
        selected_candidates = None
        selection_metadata: Dict[str, Any] = {}
        if k is not None and k > 0 and len(passed) > 0:
            if embeddings is None:
                raise ValueError("Embeddings are required for k-DPP selection.")
            selected_candidates, selection_metadata = self.select_diverse_rois(
                passed,
                k=k,
                sigma=sigma,
                tau=tau,
                iou_threshold=float(iou_threshold),
                recent_rois=recent_rois,
                recent_embeds=recent_embeds,
                recent_group_ids=recent_group_ids,
                novelty_alpha=novelty_alpha,
                novelty_epsilon=novelty_epsilon,
                iou_prev_max=iou_prev_max,
                enforce_group_spread=enforce_group_spread,
                seed=seed,
            )

        # Build metadata
        thresholds_used = self._last_qc_thresholds or {
            "tissue_threshold": min_tissue_frac,
            "artifact_threshold": max_artifact_frac,
            "tissue_percentile": None,
            "artifact_percentile": None,
        }
        metadata = {
            "n_candidates_sampled": len(candidates),
            "target_passed": target_passed,
            "max_attempts": max_attempts,
            "n_candidates_passed": len(passed),
            "n_candidates_rejected": len(rejected),
            "rejection_reasons": {
                "low_tissue": sum(1 for c in rejected if c.reject_reason == "low_tissue"),
                "high_artifact": sum(1 for c in rejected if c.reject_reason == "high_artifact"),
            },
            "qc_thresholds": {
                "min_tissue_frac": min_tissue_frac,
                "max_artifact_frac": max_artifact_frac,
                "tile_min_tissue_for_valid": tile_min_tissue_for_valid,
                "group_cover_min": group_cover_min,
                "tile_min_tissue_for_group": tile_min_tissue_for_group,
                "merge_radius_tiles": merge_radius_tiles,
                "min_group_tiles_frac": min_group_tiles_frac,
                "max_artifact_frac_for_group": max_artifact_frac_for_group,
                "min_valid_frac": min_valid_frac,
                "tissue_percentile": tissue_percentile,
                "artifact_percentile": artifact_percentile,
            },
            "qc_thresholds_used": thresholds_used,
            "tissue_groups": {
                "usable_groups": len(self._group_info) if self._group_info is not None else 0,
            },
            "grid_shape": self.grid_shape,
            "roi_size": self.roi_size,
            "has_embeddings": embeddings is not None,
            "embedding_dim": embeddings.shape[1] if embeddings is not None and len(embeddings) > 0 else None,
            "selection": selection_metadata,
        }

        return ROISamplerResult(
            candidates=candidates,
            passed_candidates=passed,
            rejected_candidates=rejected,
            embeddings=embeddings,
            selected_candidates=selected_candidates,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def sample_candidate_rois_selective(
    preprocess_dir: Union[str, Path],
    *,
    qc_zarr_path: Optional[Union[str, Path]] = None,
    features_path: Optional[Union[str, Path]] = None,
    roi_size: int = ROI_SIZE,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    target_passed: Optional[int] = DEFAULT_TARGET_PASSED,
    max_attempts: Optional[int] = None,
    min_tissue_frac: float = DEFAULT_MIN_TISSUE_FRAC,
    max_artifact_frac: float = DEFAULT_MAX_ARTIFACT_FRAC,
    tile_min_tissue_for_valid: float = DEFAULT_TILE_MIN_TISSUE_FOR_VALID,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
    group_cover_min: float = DEFAULT_GROUP_COVER_MIN,
    tile_min_tissue_for_group: float = DEFAULT_TILE_MIN_TISSUE_FOR_GROUP,
    merge_radius_tiles: int = DEFAULT_MERGE_RADIUS_TILES,
    min_group_tiles_frac: float = DEFAULT_MIN_GROUP_TILES_FRAC,
    max_artifact_frac_for_group: float = DEFAULT_MAX_ARTIFACT_FRAC_FOR_GROUP,
    tissue_percentile: Optional[float] = None,
    artifact_percentile: Optional[float] = None,
    k: Optional[int] = DEFAULT_K_SELECTED,
    sigma: Optional[float] = None,
    tau: Optional[float] = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    recent_rois: Optional[List[Tuple[int, int]]] = None,
    recent_embeds: Optional[np.ndarray] = None,
    recent_group_ids: Optional[List[int]] = None,
    novelty_alpha: float = DEFAULT_NOVELTY_ALPHA,
    novelty_epsilon: float = DEFAULT_NOVELTY_EPSILON,
    iou_prev_max: float = DEFAULT_IOU_PREV_MAX,
    enforce_group_spread: bool = False,
    device: str = "cuda",
    batch_size: int = 32,
    overwrite_features: bool = False,
    seed: Optional[int] = None,
) -> ROISamplerResult:
    """
    ROI-first sampling: gate ROIs using QC only, encode tiles for passed ROIs,
    then compute embeddings and run k-DPP selection.
    """
    preprocess_dir = Path(preprocess_dir)
    if qc_zarr_path is None:
        qc_zarr_path = preprocess_dir / "qc_grids.zarr"
    if features_path is None:
        features_path = preprocess_dir / "features.zarr"
    features_path = Path(features_path)

    sampler = ROISampler(
        qc_zarr_path=qc_zarr_path,
        features_path=None,
        roi_size=int(roi_size),
    )

    result = sampler.sample(
        n_candidates=n_candidates,
        target_passed=target_passed,
        max_attempts=max_attempts,
        min_tissue_frac=min_tissue_frac,
        max_artifact_frac=max_artifact_frac,
        tile_min_tissue_for_valid=tile_min_tissue_for_valid,
        min_valid_frac=min_valid_frac,
        group_cover_min=group_cover_min,
        tile_min_tissue_for_group=tile_min_tissue_for_group,
        merge_radius_tiles=merge_radius_tiles,
        min_group_tiles_frac=min_group_tiles_frac,
        max_artifact_frac_for_group=max_artifact_frac_for_group,
        tissue_percentile=tissue_percentile,
        artifact_percentile=artifact_percentile,
        k=0,
        sigma=sigma,
        tau=tau,
        iou_threshold=iou_threshold,
        seed=seed,
    )

    if len(result.passed_candidates) == 0:
        return result

    from pipeline.conch_encoder import write_conch_features_zarr_for_rois

    features_path, encode_stats = write_conch_features_zarr_for_rois(
        preprocess_dir,
        result.passed_candidates,
        roi_size=int(roi_size),
        output_dir=features_path.parent,
        zarr_name=Path(features_path).name,
        tile_min_tissue_for_valid=tile_min_tissue_for_valid,
        device=device,
        batch_size=int(batch_size),
        overwrite=bool(overwrite_features),
        consolidate_metadata=False,
    )

    sampler.features_path = Path(features_path)
    sampler._load_features()
    result.embeddings = sampler.compute_roi_embeddings(result.passed_candidates)

    selection_meta: Dict[str, Any] = {}
    if k is not None and k > 0 and len(result.passed_candidates) > 0:
        if result.embeddings is None:
            raise ValueError("Embeddings are required for k-DPP selection.")
        selected, selection_meta = sampler.select_diverse_rois(
            result.passed_candidates,
            k=int(k),
            sigma=sigma,
            tau=tau,
            iou_threshold=float(iou_threshold),
            recent_rois=recent_rois,
            recent_embeds=recent_embeds,
            recent_group_ids=recent_group_ids,
            novelty_alpha=novelty_alpha,
            novelty_epsilon=novelty_epsilon,
            iou_prev_max=iou_prev_max,
            enforce_group_spread=enforce_group_spread,
            seed=seed,
        )
        result.selected_candidates = selected

    result.metadata["selection"] = selection_meta
    result.metadata["encoding"] = encode_stats
    result.metadata["features_path"] = str(features_path)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


