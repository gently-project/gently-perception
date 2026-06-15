"""
OfflineSource — discover TIFF volumes on disk and yield (embryo_id, timepoint, path).

Lifted from benchmark/testset.py:OfflineTestset but with the ground-truth
coupling removed: this module yields paths, not test cases. Ground truth
never enters the loop.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_FNAME_RE = re.compile(r"^(?P<embryo>embryo_\d+)_(?P<ts>[\d_]+)\.(?:tif|tiff|npz)$", re.IGNORECASE)
_TPOINT_RE = re.compile(r"^t(?P<ts>\d+)\.(?:tif|tiff|npz)$", re.IGNORECASE)


class OfflineSource:
    """Iterate volume files in (embryo_id, timepoint) order.

    Filenames encode a wall-clock timestamp (e.g. embryo_1_20251222_175656.tif),
    not a sequential index. We sort per-embryo by that timestamp and assign
    sequential timepoint indices 0..N — matching the GT transition format.
    """

    def __init__(self, volumes_dir: Path) -> None:
        self.volumes_dir = Path(volumes_dir)
        self._items = self._discover()

    def _discover(self) -> list[tuple[str, int, Path]]:
        by_embryo: dict[str, list[tuple[str, Path]]] = {}
        for p in sorted(self.volumes_dir.rglob("*")):
            if not p.is_file():
                continue
            m = _FNAME_RE.match(p.name)
            if m:
                by_embryo.setdefault(m["embryo"], []).append((m["ts"], p))
                continue
            m = _TPOINT_RE.match(p.name)
            if m and p.parent.name.startswith("embryo_"):
                by_embryo.setdefault(p.parent.name, []).append((m["ts"], p))
        items: list[tuple[str, int, Path]] = []
        for eid, files in by_embryo.items():
            files.sort(key=lambda x: x[0])
            for t, (_, path) in enumerate(files):
                items.append((eid, t, path))
        items.sort(key=lambda x: (x[0], x[1]))
        return items

    @property
    def embryo_ids(self) -> list[str]:
        return sorted({e for e, _, _ in self._items})

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[tuple[str, int, Path]]:
        return iter(self._items)
