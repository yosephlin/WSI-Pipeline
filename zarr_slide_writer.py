"""
Zarr slide writer for grid-aligned QC channels and feature embeddings.

Intended for tile-grid outputs such as (H, W, C) CONCH features.
For CONCH v1.5 backbone features, use C=768 to match TITAN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import zarr

__all__ = ["ZarrSlideWriter"]


class ZarrSlideWriter:
    """Write per-tile QC channels and feature embeddings to a Zarr store."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        *,
        store_name: str = "features.zarr",
        overwrite: bool = True,
        feature_dtype: str = "float16",
        qc_dtype: str = "float16",
        valid_dtype: str = "uint8",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.store_name = str(store_name)
        self.overwrite = bool(overwrite)
        self.feature_dtype = np.dtype(feature_dtype)
        self.qc_dtype = np.dtype(qc_dtype)
        self.valid_dtype = np.dtype(valid_dtype)

        self.zarr_path = self.output_dir / self.store_name
        self.root: Optional[zarr.Group] = None
        self.features: Optional[zarr.Array] = None
        self.tissue_frac: Optional[zarr.Array] = None
        self.artifact_frac: Optional[zarr.Array] = None
        self.valid_mask: Optional[zarr.Array] = None
        self.grid_shape: Optional[Tuple[int, int]] = None
        self.feature_dim: Optional[int] = None

    def open(
        self,
        slide_id: str,
        H: int,
        W: int,
        C: int,
        *,
        chunks: Optional[Tuple[int, int, int]] = None,
        stride_px_lv0: Optional[int] = None,
    ) -> None:
        if self.root is not None:
            raise RuntimeError("ZarrSlideWriter is already open.")

        h = int(H)
        w = int(W)
        c = int(C)
        if h <= 0 or w <= 0 or c <= 0:
            raise ValueError(f"Invalid grid shape: H={h}, W={w}, C={c}")

        if chunks is None:
            chunks = (16, 16, c)
        if len(chunks) != 3:
            raise ValueError("chunks must be a 3-tuple (H, W, C).")
        chunk_h, chunk_w, chunk_c = (int(ch) for ch in chunks)
        if chunk_c <= 0:
            chunk_c = c

        self.output_dir.mkdir(parents=True, exist_ok=True)
        mode = "w" if self.overwrite else "a"
        root = zarr.open(str(self.zarr_path), mode=mode)

        self.features = root.create_array(
            "features",
            shape=(h, w, c),
            chunks=(chunk_h, chunk_w, chunk_c),
            dtype=self.feature_dtype,
            fill_value=np.asarray(0, dtype=self.feature_dtype),
        )
        self.tissue_frac = root.create_array(
            "tissue_frac",
            shape=(h, w),
            chunks=(chunk_h, chunk_w),
            dtype=self.qc_dtype,
            fill_value=np.asarray(0, dtype=self.qc_dtype),
        )
        self.artifact_frac = root.create_array(
            "artifact_frac",
            shape=(h, w),
            chunks=(chunk_h, chunk_w),
            dtype=self.qc_dtype,
            fill_value=np.asarray(0, dtype=self.qc_dtype),
        )
        self.valid_mask = root.create_array(
            "valid_mask",
            shape=(h, w),
            chunks=(chunk_h, chunk_w),
            dtype=self.valid_dtype,
            fill_value=np.asarray(0, dtype=self.valid_dtype),
        )

        root.attrs["slide_id"] = str(slide_id)
        root.attrs["grid_shape"] = [h, w]
        root.attrs["feature_dim"] = c
        root.attrs["feature_dtype"] = str(self.feature_dtype)
        root.attrs["qc_dtype"] = str(self.qc_dtype)
        root.attrs["valid_dtype"] = str(self.valid_dtype)
        root.attrs["chunks"] = [chunk_h, chunk_w, chunk_c]
        if stride_px_lv0 is not None:
            root.attrs["stride_px_lv0"] = int(stride_px_lv0)

        self.root = root
        self.grid_shape = (h, w)
        self.feature_dim = c
        self._validate_shapes()

    @staticmethod
    def grid_index_from_lvl0(
        x_lv0: Union[int, float],
        y_lv0: Union[int, float],
        stride_px_lv0: int,
    ) -> Tuple[int, int]:
        """
        Convert level-0 pixel coordinates to grid indices.

        For a non-overlapping lattice: stride_px_lv0 == patch_size_px_lv0.
        """
        stride = int(stride_px_lv0)
        if stride <= 0:
            raise ValueError("stride_px_lv0 must be > 0.")
        row = int(float(y_lv0) // stride)
        col = int(float(x_lv0) // stride)
        return row, col

    def write_qc(
        self,
        row: int,
        col: int,
        tissue_frac: float,
        artifact_frac: float,
        *,
        valid: Optional[bool] = None,
    ) -> None:
        self._require_open()
        self._check_bounds(row, col)
        assert self.tissue_frac is not None
        assert self.artifact_frac is not None
        assert self.valid_mask is not None
        self.tissue_frac[int(row), int(col)] = np.asarray(tissue_frac, dtype=self.qc_dtype)
        self.artifact_frac[int(row), int(col)] = np.asarray(artifact_frac, dtype=self.qc_dtype)
        if valid is not None:
            self.valid_mask[int(row), int(col)] = np.asarray(1 if valid else 0, dtype=self.valid_dtype)

    def write_qc_block(
        self,
        row0: int,
        col0: int,
        tissue_block: Union[np.ndarray, Sequence[Sequence[float]]],
        artifact_block: Union[np.ndarray, Sequence[Sequence[float]]],
        *,
        valid_block: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
    ) -> None:
        """
        Write a contiguous QC block (e.g., 16x16 window) in one slice.
        """
        self._require_open()
        tissue_arr = np.asarray(tissue_block, dtype=self.qc_dtype)
        artifact_arr = np.asarray(artifact_block, dtype=self.qc_dtype)

        if tissue_arr.shape != artifact_arr.shape:
            raise ValueError("QC block shapes must match.")
        if tissue_arr.ndim != 2:
            raise ValueError(f"QC block must be 2D, got shape {tissue_arr.shape}.")

        block_h, block_w = tissue_arr.shape
        self._check_block_bounds(row0, col0, block_h, block_w)

        assert self.tissue_frac is not None
        assert self.artifact_frac is not None
        self.tissue_frac[row0 : row0 + block_h, col0 : col0 + block_w] = tissue_arr
        self.artifact_frac[row0 : row0 + block_h, col0 : col0 + block_w] = artifact_arr

        if valid_block is not None:
            assert self.valid_mask is not None
            valid_arr = np.asarray(valid_block, dtype=self.valid_dtype)
            if valid_arr.shape != tissue_arr.shape:
                raise ValueError("valid_block shape must match QC blocks.")
            self.valid_mask[row0 : row0 + block_h, col0 : col0 + block_w] = valid_arr

    def write_feat(
        self,
        row: int,
        col: int,
        feat: Union[np.ndarray, Sequence[float]],
        *,
        valid: Optional[bool] = None,
    ) -> None:
        self._require_open()
        self._check_bounds(row, col)
        assert self.features is not None
        assert self.valid_mask is not None
        if self.feature_dim is None:
            raise RuntimeError("Feature dimension not set.")
        if valid is False:
            self.valid_mask[int(row), int(col)] = np.asarray(0, dtype=self.valid_dtype)
            return
        arr = np.asarray(feat, dtype=self.feature_dtype)
        if arr.ndim != 1 or int(arr.shape[0]) != int(self.feature_dim):
            raise ValueError(
                f"Feature dim mismatch: expected {self.feature_dim}, got shape {arr.shape}"
            )
        self.features[int(row), int(col), :] = arr
        if valid is True:
            self.valid_mask[int(row), int(col)] = np.asarray(1, dtype=self.valid_dtype)

    def write_feat_block(
        self,
        row0: int,
        col0: int,
        feat_block: Union[np.ndarray, Sequence[Sequence[Sequence[float]]]],
        *,
        valid_block: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
    ) -> None:
        """
        Write a contiguous features block (e.g., 16x16xC) in one slice.
        """
        self._require_open()
        assert self.features is not None
        assert self.valid_mask is not None
        if self.feature_dim is None:
            raise RuntimeError("Feature dimension not set.")

        block = np.asarray(feat_block, dtype=self.feature_dtype)
        if block.ndim != 3:
            raise ValueError(f"Feature block must be 3D, got shape {block.shape}.")
        block_h, block_w, block_c = block.shape
        if int(block_c) != int(self.feature_dim):
            raise ValueError(
                f"Feature dim mismatch: expected {self.feature_dim}, got {block.shape}"
            )
        self._check_block_bounds(row0, col0, block_h, block_w)
        if valid_block is not None:
            valid_arr = np.asarray(valid_block, dtype=self.valid_dtype)
            if valid_arr.shape != (block_h, block_w):
                raise ValueError("valid_block shape must match feature block spatial shape.")
            valid_bool = valid_arr.astype(bool, copy=False)
            if not np.all(valid_bool):
                block = block.copy()
                block[~valid_bool] = np.asarray(0, dtype=self.feature_dtype)
            self.valid_mask[row0 : row0 + block_h, col0 : col0 + block_w] = valid_arr

        self.features[row0 : row0 + block_h, col0 : col0 + block_w, :] = block

    def finalize(self, *, consolidate_metadata: bool = True) -> None:
        self._require_open()
        if consolidate_metadata:
            zarr.consolidate_metadata(str(self.zarr_path))
        self.root = None
        self.features = None
        self.tissue_frac = None
        self.artifact_frac = None
        self.valid_mask = None

    def _require_open(self) -> None:
        if self.root is None:
            raise RuntimeError("ZarrSlideWriter is not open.")

    def _check_bounds(self, row: int, col: int) -> None:
        if self.grid_shape is None:
            raise RuntimeError("Grid shape not initialized.")
        h, w = self.grid_shape
        r = int(row)
        c = int(col)
        if r < 0 or c < 0 or r >= h or c >= w:
            raise IndexError(f"Index out of bounds: row={r}, col={c}, grid={self.grid_shape}")

    def _check_block_bounds(self, row0: int, col0: int, block_h: int, block_w: int) -> None:
        if self.grid_shape is None:
            raise RuntimeError("Grid shape not initialized.")
        h, w = self.grid_shape
        r0 = int(row0)
        c0 = int(col0)
        bh = int(block_h)
        bw = int(block_w)
        if bh <= 0 or bw <= 0:
            raise ValueError(f"Invalid block shape: {bh}x{bw}")
        if r0 < 0 or c0 < 0 or r0 + bh > h or c0 + bw > w:
            raise IndexError(
                f"Block out of bounds: row0={r0}, col0={c0}, shape=({bh},{bw}), grid={self.grid_shape}"
            )

    def _validate_shapes(self) -> None:
        assert self.features is not None
        assert self.tissue_frac is not None
        assert self.artifact_frac is not None
        assert self.valid_mask is not None
        qc_shape = self.tissue_frac.shape
        if qc_shape != self.artifact_frac.shape:
            raise ValueError("artifact_frac shape must match tissue_frac.")
        if qc_shape != self.valid_mask.shape:
            raise ValueError("valid_mask shape must match tissue_frac.")
        if tuple(self.features.shape[:2]) != qc_shape:
            raise ValueError("features spatial shape must match QC grids.")
