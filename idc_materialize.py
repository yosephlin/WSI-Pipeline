from pathlib import Path
import shutil
from contextlib import contextmanager
from idc_index import IDCClient

@contextmanager
def materialize_series_to_tmp(series_uid: str, tmp_root: Path):
    tmp_root = Path(tmp_root)
    series_dir = tmp_root / series_uid
    series_dir.mkdir(parents=True, exist_ok=True)

    client = IDCClient()
    # downloads the DICOM instances for the series into series_dir
    client.download_from_selection(seriesInstanceUID=[series_uid], downloadDir=str(series_dir))

    try:
        # IMPORTANT: return the directory (your dataloader+wsidicom will treat folder as slide)
        yield series_dir
    finally:
        shutil.rmtree(series_dir, ignore_errors=True)
