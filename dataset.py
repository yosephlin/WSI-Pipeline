#!/usr/bin/env python3
"""
curate_idc_tcga_wsi_manifest.py

Curate a balanced, quality-filtered dataset of TCGA DICOM Whole Slide Microscopy (SM) series
from IDC using idc-index, and write a reproducible manifest (Parquet + JSONL) containing
source URIs + core metadata for downstream pipelines.

Tested against:
  idc-index 0.11.1
  duckdb 1.2.1

Outputs:
  - manifest.parquet
  - manifest.jsonl
  - train/val/test splits (optional): train.parquet, val.parquet, test.parquet (+ jsonl)
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from idc_index import IDCClient


DEFAULT_TCGA_COLLECTIONS = [
    "tcga_brca", "tcga_luad", "tcga_lusc", "tcga_coad", "tcga_read",
    "tcga_prad", "tcga_kirc", "tcga_lihc", "tcga_hnsc",
]

def _jsonify_value(x):
    # nulls
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None

    # numpy scalars -> python scalars
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    # ndarray -> list
    if isinstance(x, np.ndarray):
        return x.tolist()

    # pandas timestamp -> iso
    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    return x


def norm_collection_id(s: str) -> str:
    # normalize to match your earlier successful pattern
    return s.lower().replace("-", "_")


def sql_list_str(values: List[str]) -> str:
    # SQL string list: 'a','b','c'
    return ",".join([f"'{v}'" for v in values])


@dataclass(frozen=True)
class TargetSpacing:
    name: str
    mm_per_px: float


TARGETS = {
    "x40": TargetSpacing("x40", 0.00025),
    "x20": TargetSpacing("x20", 0.00050),
    "x10": TargetSpacing("x10", 0.00100),
}


def build_slide_table(client: IDCClient, collections_norm: List[str]) -> pd.DataFrame:
    """
    Returns one row per SeriesInstanceUID, including:
      - "base" VOLUME SOPInstanceUID selected as the largest (highest-res) VOLUME instance
      - "x20" VOLUME SOPInstanceUID closest to 0.00050 mm/px (if present)
    """
    coll_sql = sql_list_str(collections_norm)

    query = f"""
    WITH base_vol AS (
      SELECT
        SeriesInstanceUID,
        SOPInstanceUID AS base_SOPInstanceUID,
        PixelSpacing_0 AS base_PixelSpacing_mm,
        TotalPixelMatrixColumns AS base_cols,
        TotalPixelMatrixRows AS base_rows,
        instance_size AS base_instance_bytes,
        TransferSyntaxUID AS base_TransferSyntaxUID,
        ROW_NUMBER() OVER (
          PARTITION BY SeriesInstanceUID
          ORDER BY TotalPixelMatrixColumns DESC NULLS LAST,
                   instance_size DESC NULLS LAST
        ) AS rn
      FROM sm_instance_index
      WHERE list_contains(ImageType, 'VOLUME')
    ),
    x20_vol AS (
      SELECT
        SeriesInstanceUID,
        SOPInstanceUID AS x20_SOPInstanceUID,
        PixelSpacing_0 AS x20_PixelSpacing_mm,
        TotalPixelMatrixColumns AS x20_cols,
        TotalPixelMatrixRows AS x20_rows,
        instance_size AS x20_instance_bytes,
        TransferSyntaxUID AS x20_TransferSyntaxUID,
        ROW_NUMBER() OVER (
          PARTITION BY SeriesInstanceUID
          ORDER BY ABS(PixelSpacing_0 - 0.00050) ASC,
                   TotalPixelMatrixColumns DESC NULLS LAST
        ) AS rn_x20,
        MAX(CASE WHEN PixelSpacing_0 BETWEEN 0.00045 AND 0.00055 THEN 1 ELSE 0 END)
          OVER (PARTITION BY SeriesInstanceUID) AS has_x20_level
      FROM sm_instance_index
      WHERE list_contains(ImageType, 'VOLUME')
    )
    SELECT
      idx.collection_id,
      idx.source_DOI,
      idx.license_short_name,
      idx.PatientID,
      idx.StudyInstanceUID,
      idx.SeriesInstanceUID,
      idx.series_size_MB,
      idx.series_aws_url,
      idx.aws_bucket,
      idx.crdc_series_uuid,

      sm.min_PixelSpacing_2sf,
      sm.max_TotalPixelMatrixColumns,
      sm.max_TotalPixelMatrixRows,
      sm.ObjectiveLensPower,

      sm.primaryAnatomicStructure_CodeMeaning,
      sm.primaryAnatomicStructureModifier_CodeMeaning,
      sm.staining_usingSubstance_CodeMeaning,
      sm.tissueFixative_CodeMeaning,
      sm.embeddingMedium_CodeMeaning,
      sm.illuminationType_CodeMeaning,

      v.base_SOPInstanceUID,
      v.base_PixelSpacing_mm,
      v.base_cols,
      v.base_rows,
      v.base_instance_bytes,
      v.base_TransferSyntaxUID,

      x20.x20_SOPInstanceUID,
      x20.x20_PixelSpacing_mm,
      x20.x20_cols,
      x20.x20_rows,
      x20.has_x20_level

    FROM index idx
    JOIN sm_index sm USING (SeriesInstanceUID)
    JOIN base_vol v USING (SeriesInstanceUID)
    JOIN x20_vol x20 USING (SeriesInstanceUID)

    WHERE
      idx.Modality = 'SM'
      AND REPLACE(LOWER(idx.collection_id), '-', '_') IN ({coll_sql})
      AND v.rn = 1
      AND x20.rn_x20 = 1
      AND idx.series_aws_url IS NOT NULL
    """

    df = client.sql_query(query)

    # normalize string fields that may be NULL
    for col in [
        "primaryAnatomicStructure_CodeMeaning",
        "primaryAnatomicStructureModifier_CodeMeaning",
        "staining_usingSubstance_CodeMeaning",
        "tissueFixative_CodeMeaning",
        "embeddingMedium_CodeMeaning",
        "illuminationType_CodeMeaning",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # normalize collection_id for grouping consistency
    df["collection_id_norm"] = df["collection_id"].astype(str).map(norm_collection_id)

    return df


def apply_quality_filters(
    df: pd.DataFrame,
    target_spacing_mm: Optional[float],
    spacing_tol: float,
    objective_min: Optional[float],
    objective_max: Optional[float],
    min_cols: int,
    min_rows: int,
    min_series_mb: float,
    max_series_mb: float,
    allow_transfer_syntax: Optional[List[str]],
    max_base_spacing_mm: Optional[float] = None,
    require_x20_level: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    # basic sanity
    out = out.dropna(subset=["SeriesInstanceUID", "PatientID", "StudyInstanceUID", "base_SOPInstanceUID"])
    out = out[out["base_cols"] >= min_cols]
    out = out[out["base_rows"] >= min_rows]

    # series size bounds (cheap quality proxy + disk control)
    if min_series_mb is not None:
        out = out[out["series_size_MB"] >= float(min_series_mb)]
    if max_series_mb is not None:
        out = out[out["series_size_MB"] <= float(max_series_mb)]

    # objective lens power filter
    if objective_min is not None:
        out = out[out["ObjectiveLensPower"] >= float(objective_min)]
    if objective_max is not None:
        out = out[out["ObjectiveLensPower"] <= float(objective_max)]

    # spacing filter (prefer base_PixelSpacing_mm if present; fallback to min_PixelSpacing_2sf)
    if target_spacing_mm is not None:
        spacing = out["base_PixelSpacing_mm"].astype(float)
        ok = (spacing >= (target_spacing_mm - spacing_tol)) & (spacing <= (target_spacing_mm + spacing_tol))
        out = out[ok]

    # base spacing ceiling (x20-capable if <= ~0.00055 mm/px)
    if max_base_spacing_mm is not None:
        out = out[out["base_PixelSpacing_mm"].astype(float) <= float(max_base_spacing_mm)]

    # require actual x20-ish pyramid level
    if require_x20_level:
        out = out[out["has_x20_level"] == 1]

    # transfer syntax allowlist (optional)
    if allow_transfer_syntax:
        allow = set(allow_transfer_syntax)
        out = out[out["base_TransferSyntaxUID"].isin(allow)]

    return out.reset_index(drop=True)


def make_group_key(df: pd.DataFrame, balanced_by: str) -> pd.Series:
    if balanced_by == "collection":
        return df["collection_id_norm"].astype(str)
    if balanced_by == "collection_site":
        site = df["primaryAnatomicStructure_CodeMeaning"].astype(str)
        return df["collection_id_norm"].astype(str) + "||" + site
    if balanced_by == "collection_site_stain":
        site = df["primaryAnatomicStructure_CodeMeaning"].astype(str)
        stain = df["staining_usingSubstance_CodeMeaning"].astype(str)
        return df["collection_id_norm"].astype(str) + "||" + site + "||" + stain
    raise ValueError(f"Unknown balanced_by={balanced_by}")


def balanced_sample_unique_patients(
    df: pd.DataFrame,
    group_key: pd.Series,
    n_per_group: int,
    seed: int,
    one_slide_per_patient: bool = True,
) -> pd.DataFrame:
    """
    Balanced sampling across groups. Optionally enforce:
      - at most one slide per PatientID globally (recommended for leakage control)
      - OR at most one slide per PatientID per group
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["__group"] = group_key.values

    # shuffle within group for randomness
    df["__rand"] = rng.random(len(df))
    df = df.sort_values(["__group", "__rand"]).reset_index(drop=True)

    selected_rows = []
    used_patients_global = set()

    for g, gdf in df.groupby("__group", sort=False):
        if len(gdf) == 0:
            continue

        if one_slide_per_patient:
            # take unique patients in this group, skipping any patient already used globally
            gdf = gdf[~gdf["PatientID"].isin(used_patients_global)]
            if len(gdf) == 0:
                continue

            # choose 1 slide per patient (first after shuffling)
            gdf_one = gdf.drop_duplicates(subset=["PatientID"], keep="first")
            take = gdf_one.head(n_per_group)
            selected_rows.append(take)
            used_patients_global.update(take["PatientID"].tolist())
        else:
            take = gdf.head(n_per_group)
            selected_rows.append(take)

    if not selected_rows:
        return df.iloc[0:0].drop(columns=["__group", "__rand"])

    out = pd.concat(selected_rows, ignore_index=True)
    out = out.drop(columns=["__group", "__rand"]).reset_index(drop=True)
    return out


def group_split_train_val_test(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by patient (group_col) to avoid leakage across splits.
    val_size is fraction of the remaining train after test split.
    """
    df = df.copy()
    groups = df[group_col].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # val from train
    groups_train = train_df[group_col].astype(str).values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    tr_idx, val_idx = next(gss2.split(train_df, groups=groups_train))
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    return tr_df, val_df, test_df


def write_manifest(df: pd.DataFrame, out_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    parquet_path = out_prefix + ".parquet"
    jsonl_path = out_prefix + ".jsonl"

    # Parquet is robust and preserves types
    df.to_parquet(parquet_path, index=False)

    # JSONL needs JSON-serializable primitives
    df2 = df.copy()

    # Convert column-wise (fast enough for your scale)
    for c in df2.columns:
        df2[c] = df2[c].map(_jsonify_value)

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df2.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {jsonl_path}")

def summarize(df: pd.DataFrame, name: str) -> None:
    gb = df["series_size_MB"].sum() / 1024
    print(f"\n[{name}] slides={len(df):,}  approx_series_GB={gb:,.2f}")
    print(df["collection_id_norm"].value_counts().head(20))
    if "primaryAnatomicStructure_CodeMeaning" in df.columns:
        print("\nTop sites:")
        print(df["primaryAnatomicStructure_CodeMeaning"].value_counts().head(15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collections", type=str, default=",".join(DEFAULT_TCGA_COLLECTIONS),
                    help="Comma-separated collection ids (normalize form: tcga_brca, tcga_luad, ...).")
    ap.add_argument("--balanced_by", type=str, default="collection_site",
                    choices=["collection", "collection_site", "collection_site_stain"],
                    help="Stratification key for balancing.")
    ap.add_argument("--n_per_group", type=int, default=250,
                    help="Target slides per group bucket (after patient dedup).")
    ap.add_argument("--seed", type=int, default=0)

    # quality filters
    ap.add_argument("--target", type=str, default="none",
                    choices=["x40", "x20", "x10", "none"],
                    help="Target magnification proxy via pixel spacing (mm/px). Use 'none' to disable.")
    ap.add_argument("--spacing_tol", type=float, default=0.00003,
                    help="Absolute tolerance (mm/px) around target spacing.")
    ap.add_argument("--max_base_spacing_mm", type=float, default=0.00055,
                    help="Require base_PixelSpacing_mm <= this (x20-capable if ~0.00055).")
    ap.add_argument("--require_x20_level", action="store_true",
                    help="Require an actual pyramid level with PixelSpacing_0 ~ x20 (has_x20_level=1).")
    ap.add_argument("--objective_min", type=float, default=35.0)
    ap.add_argument("--objective_max", type=float, default=45.0)
    ap.add_argument("--min_cols", type=int, default=20000)
    ap.add_argument("--min_rows", type=int, default=20000)
    ap.add_argument("--min_series_mb", type=float, default=30.0)
    ap.add_argument("--max_series_mb", type=float, default=1500.0)

    ap.add_argument("--allow_transfer_syntax", type=str, default="",
                    help="Comma-separated TransferSyntaxUID allowlist (optional).")

    # selection constraints
    ap.add_argument("--one_slide_per_patient", action="store_true",
                    help="Enforce at most one slide per PatientID globally (strong leakage control).")

    # splitting
    ap.add_argument("--write_splits", action="store_true",
                    help="Write train/val/test manifests split by PatientID.")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_size", type=float, default=0.10)

    # outputs
    ap.add_argument("--out_dir", type=str, default="manifests")
    ap.add_argument("--out_name", type=str, default="tcga_wsi_manifest",
                    help="Base filename (without extension).")

    args = ap.parse_args()

    collections = [norm_collection_id(x.strip()) for x in args.collections.split(",") if x.strip()]
    target_spacing = None if args.target == "none" else TARGETS[args.target].mm_per_px
    allow_ts = [x.strip() for x in args.allow_transfer_syntax.split(",") if x.strip()] or None

    client = IDCClient()
    client.fetch_index("sm_index")
    client.fetch_index("sm_instance_index")

    print("Building slide table...")
    df_all = build_slide_table(client, collections)
    summarize(df_all, "raw")

    print("\nApplying quality filters...")
    df_q = apply_quality_filters(
        df_all,
        target_spacing_mm=target_spacing,
        spacing_tol=args.spacing_tol,
        objective_min=args.objective_min,
        objective_max=args.objective_max,
        min_cols=args.min_cols,
        min_rows=args.min_rows,
        min_series_mb=args.min_series_mb,
        max_series_mb=args.max_series_mb,
        allow_transfer_syntax=allow_ts,
        max_base_spacing_mm=args.max_base_spacing_mm,
        require_x20_level=args.require_x20_level,
    )
    summarize(df_q, "quality_filtered")

    print("\nBalanced sampling...")
    gkey = make_group_key(df_q, args.balanced_by)
    df_bal = balanced_sample_unique_patients(
        df_q,
        group_key=gkey,
        n_per_group=args.n_per_group,
        seed=args.seed,
        one_slide_per_patient=args.one_slide_per_patient,
    )
    summarize(df_bal, f"balanced({args.balanced_by})")

    # write main manifest
    out_prefix = os.path.join(args.out_dir, args.out_name)
    write_manifest(df_bal, out_prefix)

    if args.write_splits:
        tr, va, te = group_split_train_val_test(
            df_bal,
            group_col="PatientID",
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )
        write_manifest(tr, os.path.join(args.out_dir, args.out_name + "_train"))
        write_manifest(va, os.path.join(args.out_dir, args.out_name + "_val"))
        write_manifest(te, os.path.join(args.out_dir, args.out_name + "_test"))

        # quick overlap check
        tr_p = set(tr.PatientID)
        va_p = set(va.PatientID)
        te_p = set(te.PatientID)
        assert tr_p.isdisjoint(va_p) and tr_p.isdisjoint(te_p) and va_p.isdisjoint(te_p), "Patient leakage across splits!"

    print("\nDone.")


if __name__ == "__main__":
    main()
