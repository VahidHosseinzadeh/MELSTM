import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np

URL = "https://www.csc.kth.se/cvap/actions/"
ACTIONS = ("walking", "jogging", "running", "boxing", "handwaving", "handclapping")
SIZE = 32


def fetch(name, root):
    dst = root / name
    if not dst.exists():
        print(f"downloading {URL + name}", flush=True)
        part = dst.with_name(dst.name + ".part")
        urllib.request.urlretrieve(URL + name, part)
        part.rename(dst)
    return dst


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "data" / "kth"))
    p.add_argument("--keep_zips", action="store_true")
    a = p.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / "kth_frames.h5"
    fetch("00sequences.txt", root)
    if out.exists():
        print(f"{out} exists")
        return
    zips = [fetch(f"{act}.zip", root) for act in ACTIONS]
    part = root / "kth_frames.h5.part"
    with h5py.File(part, "w") as f, tempfile.TemporaryDirectory() as tmp:
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                names = sorted(n for n in zf.namelist() if n.endswith(".avi"))
                for n in names:
                    avi = Path(zf.extract(n, tmp))
                    video = avi.stem.removesuffix("_uncomp")
                    f.create_dataset(video, data=read_frames(avi), compression="gzip", compression_opts=1)
                    avi.unlink()
            print(f"{z.name}: {len(names)} videos", flush=True)
    part.rename(out)
    if not a.keep_zips:
        for z in zips:
            z.unlink()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
