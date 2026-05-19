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

_FNAME_RE = re.compile(r"(?P<embryo>embryo_\d+)[_-].*?(?P<t>\d+)\.(?:tif|tiff|npz)$", re.IGNORECASE)


class OfflineSource:
    """Iterate volume files in (embryo_id, timepoint) order."""

    def __init__(self, volumes_dir: Path) -> None:
        self.volumes_dir = Path(volumes_dir)
        self._items = self._discover()

    def _discover(self) -> list[tuple[str, int, Path]]:
        items: list[tuple[str, int, Path]] = []
        for p in sorted(self.volumes_dir.rglob("*")):
            if not p.is_file():
                continue
            m = _FNAME_RE.search(p.name)
            if not m:
                continue
            items.append((m["embryo"], int(m["t"]), p))
        items.sort(key=lambda x: (x[0], x[1]))
        return items

    @property
    def embryo_ids(self) -> list[str]:
        return sorted({e for e, _, _ in self._items})

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[tuple[str, int, Path]]:
        return iter(self._items)
