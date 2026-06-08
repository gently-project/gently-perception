"""OfflineSource: timestamp-named files → sequential timepoints per embryo."""
import numpy as np

from harness.io.volumes import OfflineSource


def test_timestamp_filenames_get_sequential_timepoints(tmp_path):
    # Real naming: embryo_N_YYYYMMDD_HHMMSS.tif
    names = [
        "embryo_1_20251222_180032.tif",
        "embryo_1_20251222_175656.tif",  # earlier timestamp
        "embryo_2_20251222_180000.tif",
    ]
    for n in names:
        np.savez(tmp_path / n.replace(".tif", ".npz"), np.zeros((4, 8, 8)))
    src = OfflineSource(tmp_path)
    items = list(src)
    assert items[0][:2] == ("embryo_1", 0)
    assert items[1][:2] == ("embryo_1", 1)
    assert items[2][:2] == ("embryo_2", 0)
    # 175656 sorts before 180032 → t=0 is the earlier file
    assert "175656" in items[0][2].name
    assert src.embryo_ids == ["embryo_1", "embryo_2"]
