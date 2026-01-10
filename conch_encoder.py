"""
CONCH 1.5 frozen encoder for histopathology tile featurization.

CONCH (CONtrastive learning from Captions for Histopathology) is a vision-language
foundation model trained on histopathology images. This module provides a simple
interface to load and use the frozen encoder for tile feature extraction.

Usage:
    from pipeline.conch_encoder import CONCHEncoder, extract_tile_features

    # Load encoder
    encoder = CONCHEncoder(device="cuda")

    # Extract features from tiles
    features = encoder.encode_tiles(tiles)  # (N, 768)

    # Or use convenience function
    features = extract_tile_features(tiles, device="cuda")
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import importlib.util
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import torch

__all__ = [
    "CONCHEncoder",
    "get_shared_conch_encoder",
    "extract_tile_features",
    "load_conch_model",
    "write_conch_features_zarr_from_preprocess",
    "write_conch_features_zarr_for_rois",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCH_EMBED_DIM = 768  # CONCH 1.5 ViT-B output dim
CONCH_INPUT_SIZE = 512  # Patch input size (model transform handles resize/crop)
CONCH_HF_MODEL_ID = "hf_hub:MahmoodLab/conchv1_5"  # HuggingFace repo for CONCH v1.5


def _require_cuda_device(device: str) -> "torch.device":
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CONCH encoder requires CUDA. Use device='cuda'.")
    return device


# ---------------------------------------------------------------------------
# CONCH Model Loading
# ---------------------------------------------------------------------------


def load_conch_model(
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """
    Load CONCH 1.5 model and preprocessing transform via Trident's loader.

    Requires HuggingFace authentication if downloading:
        huggingface-cli login

    Args:
        model_path: Path to local `pytorch_model_vision.bin` (optional, uses HF if None)
        device: CUDA device string (e.g., "cuda" or "cuda:0")

    Returns:
        Tuple of (model, transform)
    """
    device = _require_cuda_device(device)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    def _load_local_conch_module() -> Any:
        module_path = Path(__file__).resolve().parent / "conchv1_5.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Missing local CONCHv1.5 loader: {module_path}")
        spec = importlib.util.spec_from_file_location("pipeline_conchv1_5_local", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        from conchv1_5 import create_model_from_pretrained
    except Exception as exc:
        try:
            module = _load_local_conch_module()
            create_model_from_pretrained = module.create_model_from_pretrained
        except Exception as inner_exc:
            raise ImportError(
                "Failed to import local CONCHv1.5 loader. "
                "Ensure pipeline/conchv1_5.py exists and dependencies (timm, einops, einops_exts) are installed. "
                f"Underlying error: {inner_exc}"
            ) from exc

    checkpoint_path = str(model_path) if model_path is not None else CONCH_HF_MODEL_ID
    model, transform = create_model_from_pretrained(
        checkpoint_path=checkpoint_path,
        img_size=512,
    )

    model = model.to(device)
    model.eval()

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    return model, transform


# ---------------------------------------------------------------------------
# CONCH Encoder Class
# ---------------------------------------------------------------------------


class CONCHEncoder:
    """
    Frozen CONCH encoder for tile feature extraction.

    Provides a simple interface to encode histopathology tiles into
    768-dimensional feature vectors.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """
        Initialize the CONCH encoder.

        Args:
            model_path: Path to local checkpoint (None = load from HuggingFace)
            device: CUDA device string (e.g., "cuda" or "cuda:0")
            batch_size: Batch size for encoding
        """
        self.device = _require_cuda_device(device)
        self.batch_size = batch_size

        # Load model
        self.model, self.preprocess = load_conch_model(
            model_path=model_path,
            device=str(self.device),
        )

        self.embed_dim = CONCH_EMBED_DIM

    @torch.inference_mode()
    def encode_tiles(
        self,
        tiles: Union[np.ndarray, List[np.ndarray], "torch.Tensor"],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode tiles into feature vectors.

        Args:
            tiles: Input tiles as:
                - numpy array of shape (N, H, W, 3) uint8
                - list of numpy arrays (H, W, 3) uint8
                - torch tensor of shape (N, 3, H, W) float32 (already preprocessed)
            normalize: If True, L2-normalize the output features

        Returns:
            Feature array of shape (N, 768)
        """
        # Handle different input formats
        if isinstance(tiles, np.ndarray):
            if tiles.ndim == 3:
                tiles = [tiles]
            else:
                tiles = [tiles[i] for i in range(tiles.shape[0])]

        if isinstance(tiles, list):
            # Convert numpy tiles to tensor
            tiles_tensor = self._preprocess_tiles(tiles)
        else:
            tiles_tensor = tiles

        # Encode in batches
        all_features = []
        n_tiles = tiles_tensor.shape[0]

        for i in range(0, n_tiles, self.batch_size):
            batch = tiles_tensor[i:i + self.batch_size].to(self.device)

            # Get image features from CONCH
            features = self._extract_features(batch)
            all_features.append(features.cpu())

        features = torch.cat(all_features, dim=0)

        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)

        return features.numpy().astype(np.float32)

    def _preprocess_tiles(self, tiles: List[np.ndarray]) -> "torch.Tensor":
        """Preprocess numpy tiles to tensor."""
        from PIL import Image

        processed = []
        for tile in tiles:
            # Ensure uint8
            if tile.dtype != np.uint8:
                tile = (tile * 255).astype(np.uint8) if tile.max() <= 1.0 else tile.astype(np.uint8)

            # Convert to PIL
            pil_img = Image.fromarray(tile)

            # Apply preprocessing
            tensor = self.preprocess(pil_img)
            processed.append(tensor)

        return torch.stack(processed, dim=0)

    def _extract_features(self, batch: "torch.Tensor") -> "torch.Tensor":
        """Extract backbone (pre-projection) features from a batch of tiles."""
        model = self.model

        if hasattr(model, "forward_features"):
            features = model.forward_features(batch)
        elif hasattr(model, "visual"):
            visual = model.visual
            if hasattr(visual, "forward_features"):
                features = visual.forward_features(batch)
            else:
                features = visual(batch)
        else:
            features = model(batch)

        return self._select_backbone_features(features)

    def _select_backbone_features(self, features: Any) -> "torch.Tensor":
        """Select ViT backbone embeddings (CLS token) and validate dimension."""
        if isinstance(features, dict):
            for key in ("pre_logits", "x", "features", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
            else:
                raise ValueError("Unsupported feature dict from CONCH model.")

        if isinstance(features, (list, tuple)):
            features = features[0]

        if not isinstance(features, torch.Tensor):
            raise TypeError("CONCH features must be a torch.Tensor.")

        if features.ndim == 3:
            # Use CLS token for ViT-style outputs (B, tokens, C)
            features = features[:, 0]

        if features.ndim != 2:
            raise ValueError(f"Unexpected CONCH feature shape: {features.shape}")

        if features.shape[-1] != self.embed_dim:
            raise ValueError(
                f"CONCH backbone dim mismatch: expected {self.embed_dim}, got {features.shape[-1]}"
            )

        return features

    def encode_from_slide(
        self,
        slide: Any,
        tile_coords: List[Tuple[int, int]],
        tile_size_lvl0: int,
        read_level: int = 0,
        target_size: int = CONCH_INPUT_SIZE,
    ) -> np.ndarray:
        """
        Extract features directly from a slide at given coordinates.

        Args:
            slide: OpenSlide object
            tile_coords: List of (x, y) coordinates at level 0
            tile_size_lvl0: Tile size in level 0 pixels
            read_level: Pyramid level to read from
            target_size: Target size for resizing

        Returns:
            Feature array of shape (N, 768)
        """
        from PIL import Image

        level_ds = float(slide.level_downsamples[read_level])
        read_size = max(1, int(round(tile_size_lvl0 / level_ds)))

        all_features = []
        batch_tiles = []

        for x, y in tile_coords:
            # Read tile
            tile_pil = slide.read_region((x, y), read_level, (read_size, read_size))
            tile_rgb = tile_pil.convert("RGB")

            # Resize if needed
            if read_size != target_size:
                tile_rgb = tile_rgb.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # Preprocess
            tensor = self.preprocess(tile_rgb)
            batch_tiles.append(tensor)

            # Process batch
            if len(batch_tiles) >= self.batch_size:
                batch = torch.stack(batch_tiles, dim=0).to(self.device)
                features = self._extract_features(batch)
                all_features.append(features.cpu())
                batch_tiles = []

        # Process remaining tiles
        if batch_tiles:
            batch = torch.stack(batch_tiles, dim=0).to(self.device)
            features = self._extract_features(batch)
            all_features.append(features.cpu())

        if not all_features:
            return np.array([]).reshape(0, self.embed_dim)

        features = torch.cat(all_features, dim=0)
        features = features / features.norm(dim=-1, keepdim=True)

        return features.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Shared encoder cache
# ---------------------------------------------------------------------------


_ENCODER_CACHE: Dict[Tuple[Optional[str], str], CONCHEncoder] = {}


def get_shared_conch_encoder(
    *,
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> CONCHEncoder:
    """
    Return a cached CONCHEncoder for the current process.

    Reuses the model across slides to avoid repeated loads.
    """
    key = (str(model_path) if model_path is not None else None, str(device))
    encoder = _ENCODER_CACHE.get(key)
    if encoder is None:
        encoder = CONCHEncoder(model_path=model_path, device=device, batch_size=int(batch_size))
        _ENCODER_CACHE[key] = encoder
    else:
        encoder.batch_size = int(batch_size)
    return encoder


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def extract_tile_features(
    tiles: Union[np.ndarray, List[np.ndarray]],
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convenience function to extract features from tiles.

    Args:
        tiles: Input tiles (N, H, W, 3) uint8 or list of (H, W, 3)
        model_path: Path to CONCH checkpoint (None = HuggingFace)
        device: CUDA device string (e.g., "cuda" or "cuda:0")
        batch_size: Batch size
        normalize: L2-normalize features

    Returns:
        Feature array (N, 768)

    Example:
        >>> tiles = np.random.randint(0, 255, (100, 512, 512, 3), dtype=np.uint8)
        >>> features = extract_tile_features(tiles, device="cuda")
        >>> print(features.shape)  # (100, 768)
    """
    encoder = get_shared_conch_encoder(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
    )
    return encoder.encode_tiles(tiles, normalize=normalize)


def extract_features_from_slide(
    slide_path: Union[str, Path],
    tile_coords: List[Tuple[int, int]],
    tile_size_lvl0: int,
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    batch_size: int = 32,
    read_level: int = 0,
) -> np.ndarray:
    """
    Extract CONCH features from a WSI at specified tile coordinates.

    Args:
        slide_path: Path to WSI file
        tile_coords: List of (x, y) coordinates at level 0
        tile_size_lvl0: Tile size in level 0 pixels
        model_path: Path to CONCH checkpoint
        device: CUDA device string (e.g., "cuda" or "cuda:0")
        batch_size: Batch size
        read_level: Pyramid level to read from

    Returns:
        Feature array (N, 768)
    """
    import openslide
    from PIL import Image
    from PIL import Image
    from PIL import Image
    from PIL import Image
    from PIL import Image
    from PIL import Image

    slide = openslide.OpenSlide(str(slide_path))

    encoder = CONCHEncoder(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
    )

    features = encoder.encode_from_slide(
        slide=slide,
        tile_coords=tile_coords,
        tile_size_lvl0=tile_size_lvl0,
        read_level=read_level,
    )

    slide.close()
    return features


def _load_tile_manifest_grid(
    parquet_path: Path,
    grid_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("pyarrow is required to read tile_manifest.parquet.") from e

    table = pq.read_table(str(parquet_path))
    data = table.to_pydict()
    for key in ("grid_i", "grid_j", "x0_lvl0", "y0_lvl0"):
        if key not in data:
            raise KeyError(f"tile_manifest missing column: {key}")

    grid_i = np.asarray(data["grid_i"], dtype=np.int64)
    grid_j = np.asarray(data["grid_j"], dtype=np.int64)
    x0 = np.asarray(data["x0_lvl0"], dtype=np.int64)
    y0 = np.asarray(data["y0_lvl0"], dtype=np.int64)

    h, w = grid_shape
    x0_grid = np.full((h, w), -1, dtype=np.int64)
    y0_grid = np.full((h, w), -1, dtype=np.int64)

    for i, j, x, y in zip(grid_i, grid_j, x0, y0):
        if i < 0 or j < 0 or i >= h or j >= w:
            continue
        if x0_grid[int(i), int(j)] != -1:
            raise ValueError(f"Duplicate tile at grid ({i}, {j}) in {parquet_path}")
        x0_grid[int(i), int(j)] = int(x)
        y0_grid[int(i), int(j)] = int(y)

    return x0_grid, y0_grid


def _open_or_create_features_store(
    zarr_path: Path,
    *,
    grid_shape: Tuple[int, int],
    embed_dim: int,
    chunks: Tuple[int, int, int],
    tissue_frac: np.ndarray,
    artifact_frac: np.ndarray,
    slide_id: Optional[str],
    stride_px_lv0: Optional[int],
    overwrite: bool,
) -> Tuple["zarr.Group", "zarr.Array", "zarr.Array", "zarr.Array", "zarr.Array", bool]:
    import zarr

    zarr_path = Path(zarr_path)
    exists = zarr_path.exists()
    mode = "w" if overwrite or not exists else "a"
    root = zarr.open(str(zarr_path), mode=mode)

    h, w = grid_shape
    chunk_h, chunk_w, chunk_c = (int(c) for c in chunks)
    if chunk_c <= 0:
        chunk_c = int(embed_dim)

    created = False

    if "features" in root and not overwrite:
        features = root["features"]
        if tuple(features.shape[:2]) != (h, w) or int(features.shape[2]) != int(embed_dim):
            raise ValueError("Existing features.zarr shape does not match grid/embedding dim.")
    else:
        features = root.create_array(
            "features",
            shape=(h, w, int(embed_dim)),
            chunks=(chunk_h, chunk_w, chunk_c),
            dtype=np.float16,
            fill_value=np.asarray(0, dtype=np.float16),
        )
        created = True

    if "tissue_frac" in root and not overwrite:
        tissue_arr = root["tissue_frac"]
        if tuple(tissue_arr.shape) != (h, w):
            raise ValueError("Existing tissue_frac shape does not match grid.")
    else:
        tissue_arr = root.create_array(
            "tissue_frac",
            shape=(h, w),
            chunks=(chunk_h, chunk_w),
            dtype=np.float16,
            fill_value=np.asarray(0, dtype=np.float16),
        )
        tissue_arr[:] = tissue_frac.astype(np.float16, copy=False)
        created = True

    if "artifact_frac" in root and not overwrite:
        artifact_arr = root["artifact_frac"]
        if tuple(artifact_arr.shape) != (h, w):
            raise ValueError("Existing artifact_frac shape does not match grid.")
    else:
        artifact_arr = root.create_array(
            "artifact_frac",
            shape=(h, w),
            chunks=(chunk_h, chunk_w),
            dtype=np.float16,
            fill_value=np.asarray(0, dtype=np.float16),
        )
        artifact_arr[:] = artifact_frac.astype(np.float16, copy=False)
        created = True

    if "valid_mask" in root and not overwrite:
        valid_arr = root["valid_mask"]
        if tuple(valid_arr.shape) != (h, w):
            raise ValueError("Existing valid_mask shape does not match grid.")
    else:
        valid_arr = root.create_array(
            "valid_mask",
            shape=(h, w),
            chunks=(chunk_h, chunk_w),
            dtype=np.uint8,
            fill_value=np.asarray(0, dtype=np.uint8),
        )
        created = True

    if created:
        if slide_id is not None:
            root.attrs["slide_id"] = str(slide_id)
        root.attrs["grid_shape"] = [int(h), int(w)]
        root.attrs["feature_dim"] = int(embed_dim)
        root.attrs["feature_dtype"] = "float16"
        root.attrs["qc_dtype"] = "float16"
        root.attrs["valid_dtype"] = "uint8"
        root.attrs["chunks"] = [int(chunk_h), int(chunk_w), int(chunk_c)]
        if stride_px_lv0 is not None:
            root.attrs["stride_px_lv0"] = int(stride_px_lv0)

    return root, features, tissue_arr, artifact_arr, valid_arr, created


def write_conch_features_zarr_from_preprocess(
    preprocess_dir: Union[str, Path],
    *,
    wsi_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    zarr_name: str = "features.zarr",
    min_tissue_encode: Optional[float] = None,
    block_size: int = 16,
    batch_size: int = 32,
    device: str = "cuda",
    encoder: Optional["CONCHEncoder"] = None,
    model_path: Optional[Union[str, Path]] = None,
    consolidate_metadata: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Write grid-aligned CONCH features + QC channels to features.zarr.

    Expects preprocess_qc outputs:
      - preprocess_meta.json
      - qc_grids.zarr
      - tile_manifest.parquet
    """
    preprocess_dir = Path(preprocess_dir)
    meta_path = preprocess_dir / "preprocess_meta.json"
    qc_zarr_path = preprocess_dir / "qc_grids.zarr"
    manifest_path = preprocess_dir / "tile_manifest.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing preprocess_meta.json: {meta_path}")
    if not qc_zarr_path.exists():
        raise FileNotFoundError(f"Missing qc_grids.zarr: {qc_zarr_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing tile_manifest.parquet: {manifest_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    slide_id = str(meta.get("slide_id", preprocess_dir.name))
    tile_size_lvl0 = int(meta.get("tile_size_lvl0", 0) or 0)
    if tile_size_lvl0 <= 0:
        raise ValueError("preprocess_meta.json missing tile_size_lvl0.")

    if wsi_path is None:
        wsi_path = meta.get("wsi_path")
    if not wsi_path:
        raise ValueError("Provide --wsi or ensure preprocess_meta.json has wsi_path.")
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    if min_tissue_encode is None:
        min_tissue_encode = float(meta.get("min_tissue_frac", 0.0) or 0.0)

    try:
        import zarr
    except ImportError as e:
        raise ImportError("zarr is required to read qc_grids.zarr.") from e

    qc_root = zarr.open(str(qc_zarr_path), mode="r")
    tissue = np.asarray(qc_root["tissue_frac"], dtype=np.float32)
    artifact = np.asarray(qc_root["artifact_frac"], dtype=np.float32)

    if tissue.shape != artifact.shape:
        raise ValueError("QC grids must share the same spatial shape.")

    h, w = tissue.shape
    x0_grid, y0_grid = _load_tile_manifest_grid(manifest_path, (h, w))
    coords_valid = (x0_grid >= 0) & (y0_grid >= 0)
    encode_mask = coords_valid & (tissue >= float(min_tissue_encode))

    from zarr_slide_writer import ZarrSlideWriter

    out_dir = Path(output_dir) if output_dir is not None else preprocess_dir
    if encoder is None:
        encoder = get_shared_conch_encoder(
            model_path=model_path,
            device=device,
            batch_size=int(batch_size),
        )
    else:
        encoder.batch_size = int(batch_size)

    writer = ZarrSlideWriter(out_dir, store_name=str(zarr_name), overwrite=bool(overwrite))
    writer.open(
        slide_id=slide_id,
        H=h,
        W=w,
        C=int(encoder.embed_dim),
        chunks=(int(block_size), int(block_size), int(encoder.embed_dim)),
        stride_px_lv0=int(tile_size_lvl0),
    )

    slide = None
    try:
        from preprocess_qc import _open_wsi

        slide = _open_wsi(Path(wsi_path))
        for r0 in range(0, h, int(block_size)):
            for c0 in range(0, w, int(block_size)):
                bh = min(int(block_size), h - r0)
                bw = min(int(block_size), w - c0)

                tissue_block = tissue[r0 : r0 + bh, c0 : c0 + bw]
                artifact_block = artifact[r0 : r0 + bh, c0 : c0 + bw]
                valid_block = encode_mask[r0 : r0 + bh, c0 : c0 + bw].astype(np.uint8)

                writer.write_qc_block(
                    r0,
                    c0,
                    tissue_block,
                    artifact_block,
                    valid_block=valid_block,
                )

                if not np.any(valid_block):
                    continue

                coords: List[Tuple[int, int]] = []
                positions: List[Tuple[int, int]] = []
                for bi in range(bh):
                    for bj in range(bw):
                        if not valid_block[bi, bj]:
                            continue
                        gi = r0 + bi
                        gj = c0 + bj
                        x0 = int(x0_grid[gi, gj])
                        y0 = int(y0_grid[gi, gj])
                        if x0 < 0 or y0 < 0:
                            continue
                        coords.append((x0, y0))
                        positions.append((bi, bj))

                if not coords:
                    continue

                feats = encoder.encode_from_slide(
                    slide=slide,
                    tile_coords=coords,
                    tile_size_lvl0=int(tile_size_lvl0),
                    read_level=0,
                )

                feat_block = np.zeros((bh, bw, int(encoder.embed_dim)), dtype=np.float32)
                for (bi, bj), feat in zip(positions, feats, strict=True):
                    feat_block[int(bi), int(bj), :] = feat

                writer.write_feat_block(
                    r0,
                    c0,
                    feat_block,
                    valid_block=valid_block,
                )
    finally:
        if slide is not None:
            slide.close()
        writer.finalize(consolidate_metadata=bool(consolidate_metadata))

    return writer.zarr_path


def write_conch_features_zarr_for_rois(
    preprocess_dir: Union[str, Path],
    roi_candidates: List[Any],
    *,
    roi_size: int = 16,
    wsi_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    zarr_name: str = "features.zarr",
    tile_min_tissue_for_valid: float = 0.3,
    device: str = "cuda",
    batch_size: int = 32,
    chunk_size: int = 16,
    overwrite: bool = False,
    consolidate_metadata: bool = True,
    prefetch_batches: int = 2,
    encoder: Optional["CONCHEncoder"] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Dict[str, int]]:
    """
    Encode only tiles needed for QC-passed ROI candidates and write to features.zarr.

    Args:
        preprocess_dir: Directory containing preprocess_meta.json, qc_grids.zarr, tile_manifest.parquet
        roi_candidates: List of ROICandidate or (grid_i, grid_j) tuples (top-left ROI)
        roi_size: ROI size in tiles
        wsi_path: Optional WSI path override
        output_dir: Optional output directory (default: preprocess_dir)
        zarr_name: Zarr store name (default: features.zarr)
        tile_min_tissue_for_valid: Minimum tile tissue fraction to encode
        device: CUDA device string
        batch_size: Encoder batch size
        chunk_size: Zarr chunk size (tiles)
        overwrite: If True, overwrite features.zarr
        consolidate_metadata: Consolidate Zarr metadata on finalize
        prefetch_batches: Number of prefetch batches for CPU tile reading (0 disables)

    Returns:
        Tuple of (features_zarr_path, stats dict)
    """
    preprocess_dir = Path(preprocess_dir)
    meta_path = preprocess_dir / "preprocess_meta.json"
    qc_zarr_path = preprocess_dir / "qc_grids.zarr"
    manifest_path = preprocess_dir / "tile_manifest.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing preprocess_meta.json: {meta_path}")
    if not qc_zarr_path.exists():
        raise FileNotFoundError(f"Missing qc_grids.zarr: {qc_zarr_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing tile_manifest.parquet: {manifest_path}")

    if not roi_candidates:
        raise ValueError("roi_candidates is empty; nothing to encode.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    slide_id = str(meta.get("slide_id", preprocess_dir.name))
    tile_size_lvl0 = int(meta.get("tile_size_lvl0", 0) or 0)
    if tile_size_lvl0 <= 0:
        raise ValueError("preprocess_meta.json missing tile_size_lvl0.")

    try:
        import zarr
    except ImportError as e:
        raise ImportError("zarr is required to read qc_grids.zarr.") from e

    qc_root = zarr.open(str(qc_zarr_path), mode="r")
    tissue = np.asarray(qc_root["tissue_frac"], dtype=np.float32)
    artifact = np.asarray(qc_root["artifact_frac"], dtype=np.float32)

    if tissue.shape != artifact.shape:
        raise ValueError("QC grids must share the same spatial shape.")

    h, w = tissue.shape
    x0_grid, y0_grid = _load_tile_manifest_grid(manifest_path, (h, w))

    valid_mask = tissue >= float(tile_min_tissue_for_valid)
    coords_valid = (x0_grid >= 0) & (y0_grid >= 0)

    needed_mask = np.zeros((h, w), dtype=bool)
    roi_size = int(roi_size)
    for candidate in roi_candidates:
        if hasattr(candidate, "grid_i") and hasattr(candidate, "grid_j"):
            i = int(candidate.grid_i)
            j = int(candidate.grid_j)
        else:
            i = int(candidate[0])
            j = int(candidate[1])
        if i < 0 or j < 0 or i >= h or j >= w:
            continue
        i1 = min(h, i + roi_size)
        j1 = min(w, j + roi_size)
        needed_mask[i:i1, j:j1] = True

    needed_mask &= valid_mask
    needed_mask &= coords_valid
    requested_tiles = int(np.count_nonzero(needed_mask))

    out_dir = Path(output_dir) if output_dir is not None else preprocess_dir
    zarr_path = out_dir / zarr_name

    existing_valid = None
    if zarr_path.exists():
        try:
            root = zarr.open(str(zarr_path), mode="r")
            if "valid_mask" in root:
                existing_valid = np.asarray(root["valid_mask"][:], dtype=bool)
        except Exception:
            existing_valid = None

    if existing_valid is not None:
        needed_mask &= ~existing_valid
        to_encode = int(np.count_nonzero(needed_mask))
        if to_encode == 0:
            if consolidate_metadata:
                zarr.consolidate_metadata(str(zarr_path))
            return zarr_path, {
                "requested_tiles": requested_tiles,
                "already_encoded": requested_tiles,
                "encoded_now": 0,
            }

    if wsi_path is None:
        wsi_path = meta.get("wsi_path")
    if not wsi_path:
        raise ValueError("Provide --wsi or ensure preprocess_meta.json has wsi_path.")
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    if encoder is None:
        encoder = get_shared_conch_encoder(
            model_path=model_path,
            device=device,
            batch_size=int(batch_size),
        )
    else:
        encoder.batch_size = int(batch_size)
    root, features_arr, tissue_arr, artifact_arr, valid_arr, _created = _open_or_create_features_store(
        zarr_path,
        grid_shape=(h, w),
        embed_dim=int(encoder.embed_dim),
        chunks=(int(chunk_size), int(chunk_size), int(encoder.embed_dim)),
        tissue_frac=tissue,
        artifact_frac=artifact,
        slide_id=slide_id,
        stride_px_lv0=tile_size_lvl0,
        overwrite=bool(overwrite),
    )

    encoded_mask = np.asarray(valid_arr[:], dtype=bool)
    needed_mask &= ~encoded_mask
    to_encode = int(np.count_nonzero(needed_mask))

    if to_encode == 0:
        if consolidate_metadata:
            zarr.consolidate_metadata(str(zarr_path))
        return zarr_path, {
            "requested_tiles": requested_tiles,
            "already_encoded": requested_tiles,
            "encoded_now": 0,
        }

    grid_coords: List[Tuple[int, int]] = []
    tile_coords: List[Tuple[int, int]] = []
    for i, j in np.argwhere(needed_mask):
        x0 = int(x0_grid[i, j])
        y0 = int(y0_grid[i, j])
        if x0 < 0 or y0 < 0:
            continue
        grid_coords.append((int(i), int(j)))
        tile_coords.append((x0, y0))

    if not tile_coords:
        if consolidate_metadata:
            zarr.consolidate_metadata(str(zarr_path))
        return zarr_path, {
            "requested_tiles": requested_tiles,
            "already_encoded": requested_tiles - to_encode,
            "encoded_now": 0,
        }

    from PIL import Image
    from preprocess_qc import _open_wsi

    # Threaded prefetch is safer with OpenSlide than wsidicom; disable for DICOM dirs.
    try:
        _is_dicom_input = Path(wsi_path).is_dir()
    except Exception:
        _is_dicom_input = False
    if _is_dicom_input:
        prefetch_batches = 0

    if int(prefetch_batches) <= 0:
        slide = None
        try:
            slide = _open_wsi(Path(wsi_path))
            feats = encoder.encode_from_slide(
                slide=slide,
                tile_coords=tile_coords,
                tile_size_lvl0=int(tile_size_lvl0),
                read_level=0,
            )
        finally:
            if slide is not None:
                slide.close()

        for (i, j), feat in zip(grid_coords, feats, strict=True):
            features_arr[int(i), int(j), :] = feat.astype(np.float16, copy=False)
            valid_arr[int(i), int(j)] = np.uint8(1)
    else:
        import queue as queue_lib
        import threading

        prefetch_limit = max(1, int(prefetch_batches))
        batch_size = max(1, int(batch_size))
        task_queue: "queue_lib.Queue[Optional[Tuple[List[Tuple[int, int]], torch.Tensor]]]" = queue_lib.Queue(
            maxsize=prefetch_limit
        )
        errors: List[BaseException] = []

        def _reader_worker() -> None:
            slide = None
            try:
                slide = _open_wsi(Path(wsi_path))
                level_ds = float(slide.level_downsamples[0])
                read_size = max(1, int(round(tile_size_lvl0 / level_ds)))
                batch_tiles: List[torch.Tensor] = []
                batch_grid: List[Tuple[int, int]] = []
                for (grid_i, grid_j), (x0, y0) in zip(grid_coords, tile_coords, strict=True):
                    tile_pil = slide.read_region((x0, y0), 0, (read_size, read_size)).convert("RGB")
                    if read_size != CONCH_INPUT_SIZE:
                        tile_pil = tile_pil.resize(
                            (CONCH_INPUT_SIZE, CONCH_INPUT_SIZE),
                            Image.Resampling.LANCZOS,
                        )
                    batch_tiles.append(encoder.preprocess(tile_pil))
                    batch_grid.append((grid_i, grid_j))
                    if len(batch_tiles) >= batch_size:
                        task_queue.put((batch_grid, torch.stack(batch_tiles, dim=0)))
                        batch_tiles = []
                        batch_grid = []
                if batch_tiles:
                    task_queue.put((batch_grid, torch.stack(batch_tiles, dim=0)))
            except BaseException as exc:
                errors.append(exc)
            finally:
                if slide is not None:
                    slide.close()
                task_queue.put(None)

        thread = threading.Thread(target=_reader_worker, daemon=True)
        thread.start()

        with torch.inference_mode():
            while True:
                item = task_queue.get()
                if item is None:
                    break
                batch_grid, batch_tiles = item
                batch = batch_tiles.to(encoder.device)
                features = encoder._extract_features(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                features_cpu = features.cpu().numpy().astype(np.float32)
                for (i, j), feat in zip(batch_grid, features_cpu, strict=True):
                    features_arr[int(i), int(j), :] = feat.astype(np.float16, copy=False)
                    valid_arr[int(i), int(j)] = np.uint8(1)

        thread.join()
        if errors:
            raise RuntimeError(f"Prefetch worker failed: {errors[0]}") from errors[0]

    if consolidate_metadata:
        zarr.consolidate_metadata(str(zarr_path))

    return zarr_path, {
        "requested_tiles": requested_tiles,
        "already_encoded": requested_tiles - len(grid_coords),
        "encoded_now": len(grid_coords),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CONCH features from tiles or slides.")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test CONCH loading")
    test_parser.add_argument("--device", type=str, default="cuda", help="CUDA device string")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract features from slide")
    extract_parser.add_argument("slide", type=str, help="Path to WSI")
    extract_parser.add_argument("--output", "-o", type=str, required=True, help="Output .npy path")
    extract_parser.add_argument("--coords", type=str, help="Path to coords .npy file")
    extract_parser.add_argument("--tile-size", type=int, default=512, help="Tile size at level 0")
    extract_parser.add_argument("--device", type=str, default="cuda", help="CUDA device string")
    extract_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Zarr writer command
    zarr_parser = subparsers.add_parser("write-zarr", help="Write grid-aligned features.zarr from preprocess_qc.")
    zarr_parser.add_argument("--preprocess-dir", type=str, required=True, help="Path to preprocess_qc output dir")
    zarr_parser.add_argument("--wsi", type=str, default=None, help="Path to WSI (overrides meta)")
    zarr_parser.add_argument("--output-dir", type=str, default=None, help="Output directory for Zarr")
    zarr_parser.add_argument("--zarr-name", type=str, default="features.zarr", help="Zarr store name")
    zarr_parser.add_argument("--min-tissue-encode", type=float, default=None, help="Min tissue for encoding")
    zarr_parser.add_argument("--block-size", type=int, default=16, help="Block size for Zarr writes")
    zarr_parser.add_argument("--device", type=str, default="cuda", help="CUDA device string")
    zarr_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    zarr_parser.add_argument("--no-consolidate", action="store_true", help="Skip metadata consolidation")
    zarr_parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing Zarr store")

    args = parser.parse_args()

    if args.command == "test":
        print(f"Loading CONCH on {args.device}...")
        encoder = CONCHEncoder(device=args.device)
        print(f"  Model loaded successfully")
        print(f"  Embedding dim: {encoder.embed_dim}")
        print(f"  Device: {encoder.device}")

        # Test with random input
        dummy = np.random.randint(0, 255, (4, 512, 512, 3), dtype=np.uint8)
        features = encoder.encode_tiles(dummy)
        print(f"  Test encoding: {dummy.shape} -> {features.shape}")

    elif args.command == "extract":
        print(f"Extracting features from: {args.slide}")

        if args.coords:
            coords = np.load(args.coords)
            tile_coords = [(int(c[0]), int(c[1])) for c in coords]
        else:
            raise ValueError("--coords required for extraction")

        features = extract_features_from_slide(
            slide_path=args.slide,
            tile_coords=tile_coords,
            tile_size_lvl0=args.tile_size,
            device=args.device,
            batch_size=args.batch_size,
        )

        np.save(args.output, features)
        print(f"Saved features: {features.shape} -> {args.output}")

    elif args.command == "write-zarr":
        out_path = write_conch_features_zarr_from_preprocess(
            preprocess_dir=args.preprocess_dir,
            wsi_path=args.wsi,
            output_dir=args.output_dir,
            zarr_name=args.zarr_name,
            min_tissue_encode=args.min_tissue_encode,
            block_size=args.block_size,
            batch_size=args.batch_size,
            device=args.device,
            consolidate_metadata=not bool(args.no_consolidate),
            overwrite=not bool(args.no_overwrite),
        )
        print(f"Wrote features Zarr: {out_path}")

    else:
        parser.print_help()
