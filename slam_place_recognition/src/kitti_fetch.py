"""HTTP Range access to remote ZIP archives + KITTI / SemanticKITTI fetching.

Why
---
``data_odometry_velodyne.zip`` is 84.8 GB but we only need
``sequences/05/velodyne/*.bin`` (~5.2 GB) and possibly only a subset of frames.
Both the KITTI S3 bucket and semantic-kitti.org advertise ``Accept-Ranges:
bytes``, so the central directory can be read and individual members can be
downloaded with byte-range requests - no full download required.

:class:`HTTPRangeFile` is a seekable, thread-safe, block-cached file-like object;
:class:`zipfile.ZipFile` can be used on it directly.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

KITTI_S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
VELODYNE_ZIP = f"{KITTI_S3}/data_odometry_velodyne.zip"
POSES_ZIP = f"{KITTI_S3}/data_odometry_poses.zip"
LABELS_ZIP = "http://semantic-kitti.org/assets/data_odometry_labels.zip"


# --------------------------------------------------------------------------- #
def http_size(url: str, timeout: float = 60.0) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers["Content-Length"])


def _fetch_range(url: str, start: int, end: int, timeout: float = 120.0,
                 retries: int = 5) -> bytes:
    """Inclusive byte range [start, end]."""

    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}",
                              "User-Agent": "jsss-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) != (end - start + 1):
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            return data
        except Exception as exc:                       # noqa: BLE001
            last = exc
            time.sleep(min(2 ** attempt, 15))
    raise IOError(f"range fetch failed [{start}, {end}]: {last}")


# --------------------------------------------------------------------------- #
class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file using cached byte ranges.

    Blocks are fetched on demand and shared across threads (a per-block lock
    prevents duplicate downloads).  Suitable for ``zipfile.ZipFile``.
    """

    def __init__(self, url: str, block_size: int = 4 << 20,
                 cache_blocks: int = 64, timeout: float = 120.0):
        self.url = url
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.timeout = timeout
        self.size = http_size(url, timeout=timeout)
        self.pos = 0
        self._cache: "Dict[int, bytes]" = {}
        self._order: List[int] = []
        self._lock = threading.Lock()
        self._block_locks: Dict[int, threading.Lock] = {}
        self.bytes_downloaded = 0
        self.requests = 0

    # ------------------------------------------------------------------ #
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    # ------------------------------------------------------------------ #
    def _block(self, idx: int) -> bytes:
        with self._lock:
            got = self._cache.get(idx)
            if got is not None:
                return got
            blk_lock = self._block_locks.setdefault(idx, threading.Lock())
        with blk_lock:
            with self._lock:
                got = self._cache.get(idx)
                if got is not None:
                    return got
            start = idx * self.block_size
            end = min(start + self.block_size, self.size) - 1
            data = _fetch_range(self.url, start, end, timeout=self.timeout)
            with self._lock:
                self._cache[idx] = data
                self.bytes_downloaded += len(data)
                self.requests += 1
                self._order.append(idx)
                while len(self._order) > self.cache_blocks:
                    old = self._order.pop(0)
                    self._cache.pop(old, None)
                    self._block_locks.pop(old, None)
            return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        while n > 0:
            idx = self.pos // self.block_size
            off = self.pos % self.block_size
            blk = self._block(idx)
            take = min(n, len(blk) - off)
            if take <= 0:                     # past EOF of the last block
                break
            out += blk[off:off + take]
            self.pos += take
            n -= take
        return bytes(out)

    def readinto(self, b) -> int:             # pragma: no cover
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def stats(self) -> str:
        return (f"{self.bytes_downloaded/1e6:.1f} MB in {self.requests} range "
                f"requests")


# --------------------------------------------------------------------------- #
def download_small(url: str, dest: str, progress: bool = True) -> str:
    """Stream a small file to disk."""

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "jsss-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if progress and total:
                pct = 100.0 * done / total
                print(f"\r    {os.path.basename(dest)}: {pct:5.1f}% "
                      f"({done/1e6:.0f}/{total/1e6:.0f} MB)", end="", flush=True)
    if progress:
        print()
    os.replace(tmp, dest)
    return dest


def extract_zip_members(zip_path: str, dest_root: str, prefix: str = "",
                        members: Optional[Iterable[str]] = None,
                        strip_prefix: str = "") -> List[str]:
    """Extract selected members of a *local* zip file."""

    written = []
    with zipfile.ZipFile(zip_path) as zf:
        names = members or [n for n in zf.namelist() if n.startswith(prefix)]
        for name in names:
            if name.endswith("/"):
                continue
            rel = name[len(strip_prefix):] if name.startswith(strip_prefix) else name
            out = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with zf.open(name) as src, open(out, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            written.append(out)
    return written


# --------------------------------------------------------------------------- #
def remote_members(url: str, prefix: str = "",
                   cache_blocks: int = 64) -> Tuple[HTTPRangeFile, List[zipfile.ZipInfo]]:
    """Open a remote zip and list its members (central directory only)."""

    rf = HTTPRangeFile(url, cache_blocks=cache_blocks)
    zf = zipfile.ZipFile(rf)
    infos = [i for i in zf.infolist() if i.filename.startswith(prefix)]
    return rf, infos


def fetch_remote_members(url: str, names: Iterable[str], dest_root: str,
                         threads: int = 8, verbose: bool = True,
                         compress_type_ok: bool = True) -> List[str]:
    """Download selected members of a remote zip using range requests.

    One :class:`HTTPRangeFile` (shared block cache) feeds one
    ``zipfile.ZipFile`` per worker thread.
    """

    names = list(names)
    if not names:
        return []
    rf = HTTPRangeFile(url, cache_blocks=128)
    zf_main = zipfile.ZipFile(rf)
    info = {i.filename: i for i in zf_main.infolist()}
    missing = [n for n in names if n not in info]
    if missing:
        raise KeyError(f"{len(missing)} members not found, e.g. {missing[:3]}")

    done = [0]
    failed: List[Tuple[str, str]] = []
    lock = threading.Lock()
    t0 = time.time()

    def worker(subset: List[str]):
        local = zipfile.ZipFile(rf)          # shares the cached range file
        for name in subset:
            out = os.path.join(dest_root, name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out) and os.path.getsize(out) == info[name].file_size:
                with lock:
                    done[0] += 1
                continue
            try:
                with local.open(name) as src, open(out + ".part", "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                os.replace(out + ".part", out)
            except Exception as exc:          # noqa: BLE001
                failed.append((name, str(exc)))
            with lock:
                done[0] += 1
                if verbose and (done[0] % 25 == 0 or done[0] == len(names)):
                    el = time.time() - t0
                    rate = rf.bytes_downloaded / max(el, 1e-9) / 1e6
                    eta = (len(names) - done[0]) * el / max(done[0], 1)
                    print(f"\r    {done[0]}/{len(names)} members  "
                          f"{rf.bytes_downloaded/1e9:.2f} GB  "
                          f"{rate:.1f} MB/s  eta {eta/60:.1f} min",
                          end="", flush=True)

    chunks = [names[i::threads] for i in range(threads)]
    threads_obj = [threading.Thread(target=worker, args=(c,), daemon=True)
                   for c in chunks if c]
    for t in threads_obj:
        t.start()
    for t in threads_obj:
        t.join()
    if verbose:
        print()
    if failed:
        print(f"  WARNING: {len(failed)} member(s) failed, e.g. {failed[:3]}",
              file=sys.stderr)
    return [os.path.join(dest_root, n) for n in names
            if os.path.exists(os.path.join(dest_root, n))]
