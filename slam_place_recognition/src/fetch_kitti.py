"""Download KITTI Odometry (+ SemanticKITTI labels) into ``dat/``.

Only the needed bytes are transferred: the 84.8 GB velodyne archive is read as a
remote zip and individual members are fetched with HTTP range requests.

    python fetch_kitti.py --sequence 05                  # full sequence 05 (~5.5 GB)
    python fetch_kitti.py --sequence 05 --frames 1400    # every other frame (~2.8 GB)
    python fetch_kitti.py --sequence 05 --labels-only
    python fetch_kitti.py --posese-only

Typical aggregate throughput is ~8 MB/s with 24-32 parallel connections
(single-connection S3 throughput is throttled to ~0.3 MB/s from many networks).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DAT_DIR  # noqa: E402
from kitti_fetch import (  # noqa: E402
    LABELS_ZIP,
    POSES_ZIP,
    VELODYNE_ZIP,
    HTTPRangeFile,
    download_small,
    extract_zip_members,
)

KITTI_ROOT = os.path.join(DAT_DIR, "kitti")
SEMKITTI_ROOT = os.path.join(DAT_DIR, "semantickitti")


# --------------------------------------------------------------------------- #
def fetch_poses(dest_root: str = KITTI_ROOT) -> str:
    url = POSES_ZIP
    tmp = os.path.join("/tmp", "data_odometry_poses.zip")
    print(f"[poses] downloading {url}")
    download_small(url, tmp)
    out = extract_zip_members(tmp, dest_root)
    print(f"[poses] extracted {len(out)} file(s) -> {dest_root}/dataset/poses/")
    os.remove(tmp)
    return os.path.join(dest_root, "dataset", "poses")


# --------------------------------------------------------------------------- #
def fetch_labels(sequence: str, dest_root: str = SEMKITTI_ROOT,
                 threads: int = 16) -> str:
    """Download the (small) label archive and extract one sequence."""

    tmp = os.path.join("/tmp", "data_odometry_labels.zip")
    print(f"[labels] downloading {LABELS_ZIP} (~179 MB, {threads} connections)")
    _parallel_download(LABELS_ZIP, tmp, threads=threads)
    prefix = f"dataset/sequences/{sequence}/labels/"
    print(f"[labels] extracting {prefix}")
    out = extract_zip_members(tmp, dest_root, prefix=prefix)
    print(f"[labels] extracted {len(out)} label file(s) -> {dest_root}/{prefix}")
    os.remove(tmp)
    return os.path.join(dest_root, prefix)


def _parallel_download(url: str, dest: str, threads: int = 16,
                       chunk: int = 4 << 20) -> str:
    """Multi-connection range download of a whole (small) file."""

    from kitti_fetch import http_size, _fetch_range

    size = http_size(url)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.truncate(size)
    ranges = [(i, min(i + chunk, size) - 1) for i in range(0, size, chunk)]
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(part):
        with open(tmp, "r+b") as fh:
            for idx in part:
                s, e = ranges[idx]
                if e < s:
                    continue
                data = _fetch_range(url, s, e)
                fh.seek(s)
                fh.write(data)
                with lock:
                    done[0] += len(data)
                    if done[0] % (32 << 20) < chunk:
                        el = time.time() - t0
                        print(f"\r    {done[0]/1e6:.0f}/{size/1e6:.0f} MB "
                              f"({done[0]/1e6/max(el,1e-9):.1f} MB/s)",
                              end="", flush=True)

    idxs = list(range(len(ranges)))
    parts = [idxs[i::threads] for i in range(threads)]
    ts = [threading.Thread(target=work, args=(p,), daemon=True) for p in parts if p]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print()
    os.replace(tmp, dest)
    return dest


# --------------------------------------------------------------------------- #
def fetch_velodyne(sequence: str, dest_root: str = KITTI_ROOT,
                   threads: int = 24, frames: int | None = None,
                   block_mb: int = 4, stride: int | None = None) -> str:
    """Range-download ``dataset/sequences/<seq>/velodyne/*.bin``."""

    print(f"[velodyne] reading central directory of the 84.8 GB archive ...")
    t0 = time.time()
    rf = HTTPRangeFile(VELODYNE_ZIP, block_size=block_mb << 20, cache_blocks=64)
    with zipfile.ZipFile(rf) as zf:
        infos = [i for i in zf.infolist()
                 if i.filename.startswith(f"dataset/sequences/{sequence}/velodyne/")
                 and i.filename.endswith(".bin")]
    infos.sort(key=lambda i: i.filename)
    total = sum(i.file_size for i in infos)
    print(f"[velodyne] {len(infos)} frames, {total/1e9:.2f} GB uncompressed "
          f"(central directory in {time.time()-t0:.0f}s)")

    if stride and stride > 1:
        infos = infos[::stride]
    if frames and len(infos) > frames:
        keep = [round(i) for i in
                __import__("numpy").linspace(0, len(infos) - 1, frames)]
        infos = [infos[i] for i in sorted(set(keep))]
    need = sum(i.file_size for i in infos)
    print(f"[velodyne] downloading {len(infos)} frames ({need/1e9:.2f} GB) "
          f"with {threads} connections")

    dest = os.path.join(dest_root, "dataset", "sequences", sequence)
    os.makedirs(os.path.join(dest, "velodyne"), exist_ok=True)
    _fetch_members_parallel(VELODYNE_ZIP, infos, dest, threads=threads,
                            block_mb=block_mb,
                            strip_prefix=f"dataset/sequences/{sequence}/")
    return os.path.join(dest, "velodyne")


def _fetch_members_parallel(url: str, infos, dest_root: str,
                            threads: int = 24, block_mb: int = 4,
                            strip_prefix: str = "") -> None:
    """Contiguous-chunk, thread-local-cache member download.

    ``strip_prefix`` is removed from each member name before joining it with
    ``dest_root`` (without it the archive path would be duplicated).
    """

    by_offset = sorted(infos, key=lambda i: i.header_offset)
    # contiguous chunks keep each worker's block cache hot (strided chunks would
    # force one ranged read per frame)
    n = len(by_offset)
    bounds = [round(i * n / threads) for i in range(threads + 1)]
    chunks = [by_offset[bounds[i]:bounds[i + 1]] for i in range(threads)]
    chunks = [c for c in chunks if c]
    lock = threading.Lock()
    state = {"done": 0, "bytes": 0, "t0": time.time()}
    total_n = len(by_offset)
    total_b = sum(i.file_size for i in by_offset)
    failed: list = []

    def worker(subset):
        rf = HTTPRangeFile(url, block_size=block_mb << 20, cache_blocks=32)
        zf = zipfile.ZipFile(rf)
        for info in subset:
            rel = info.filename[len(strip_prefix):] \
                if info.filename.startswith(strip_prefix) else info.filename
            out = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out) and os.path.getsize(out) == info.file_size:
                with lock:
                    state["done"] += 1
                continue
            ok = False
            last = None
            for attempt in range(3):
                try:
                    with zf.open(info) as src, open(out + ".part", "wb") as dst:
                        while True:
                            chunk = src.read(1 << 20)
                            if not chunk:
                                break
                            dst.write(chunk)
                    if os.path.getsize(out + ".part") == info.file_size:
                        os.replace(out + ".part", out)
                        ok = True
                        break
                except Exception as exc:            # noqa: BLE001
                    last = exc
                    rf = HTTPRangeFile(url, block_size=block_mb << 20,
                                       cache_blocks=32)
                    zf = zipfile.ZipFile(rf)
                    time.sleep(1.5 * (attempt + 1))
            if not ok:
                failed.append((info.filename, str(last)))
            with lock:
                state["done"] += 1
                state["bytes"] += info.file_size
                if state["done"] % 50 == 0 or state["done"] == total_n:
                    el = time.time() - state["t0"]
                    rate = state["bytes"] / max(el, 1e-9)      # bytes/s
                    eta = (total_b - state["bytes"]) / max(rate, 1e-9) / 60.0
                    print(f"\r    {state['done']:5d}/{total_n} frames  "
                          f"{state['bytes']/1e9:5.2f}/{total_b/1e9:.2f} GB  "
                          f"{rate/1e6:5.2f} MB/s  eta {eta:4.1f} min",
                          end="", flush=True)

    ts = [threading.Thread(target=worker, args=(c,), daemon=True) for c in chunks]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print()
    if failed:
        print(f"  WARNING: {len(failed)} frame(s) failed: {failed[:3]}",
              file=sys.stderr)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="05")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--frames", type=int, default=None,
                    help="keep at most N evenly spaced frames")
    ap.add_argument("--stride", type=int, default=None,
                    help="keep every k-th frame")
    ap.add_argument("--block-mb", type=int, default=4)
    ap.add_argument("--poses-only", action="store_true")
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--no-poses", action="store_true")
    args = ap.parse_args(argv)

    if args.poses_only:
        fetch_poses()
        return
    if args.labels_only:
        fetch_labels(args.sequence)
        return

    if not args.no_poses:
        fetch_poses()
    if not args.no_labels:
        try:
            fetch_labels(args.sequence, threads=min(args.threads, 16))
        except Exception as exc:                     # noqa: BLE001
            print(f"  labels failed ({exc}); continuing without labels",
                  file=sys.stderr)
    fetch_velodyne(args.sequence, threads=args.threads, frames=args.frames,
                   stride=args.stride, block_mb=args.block_mb)
    print("\ndone. verify with:  python prepare_data.py --check kitti")


if __name__ == "__main__":
    main()
