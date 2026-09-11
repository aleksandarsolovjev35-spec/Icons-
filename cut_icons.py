#!/usr/bin/env python3
"""Cut RPG Loot Icons sheets into individual icons.

Each "RPG Loot Icons NN" folder contains 2 sheets ("Part 1/2", 1800x1013 JPG,
10 cols x 5 rows = 50 icons each). This script detects the grid of every
sheet automatically (grid position varies slightly between sheets; folders
39-40 have no cell borders) and saves 100 PNG icons per folder:

    RPG Loot Icons NN/icons/icon_001.png ... icon_100.png
    (Part 1 -> 001-050, Part 2 -> 051-100, row-major order)

Icons are 148x148, cropped centered on each detected cell, without the thin
gray cell border, on the original black background.

Usage:
    python3 cut_icons.py            # process all folders
    python3 cut_icons.py 01         # process only folders matching "01"

Requires: Pillow (pip install pillow). No numpy/cv2 needed.
"""
import glob
import os
import sys

from PIL import Image

SIZE = 148  # uniform output icon size (px), centered on detected cell
BG_LEVEL = 25  # pixel brightness threshold: <= this counts as black background
STEP = 2  # sampling step for profiles (speed; detection is robust to it)


def profiles(path):
    """Column/row counts of non-black pixels."""
    im = Image.open(path).convert("L")
    w, h = im.size
    px = im.load()
    col = [0] * w
    for x in range(w):
        c = 0
        for y in range(0, h, STEP):
            if px[x, y] > BG_LEVEL:
                c += 1
        col[x] = c
    row = [0] * h
    for y in range(h):
        c = 0
        for x in range(0, w, STEP):
            if px[x, y] > BG_LEVEL:
                c += 1
        row[y] = c
    return col, row, w, h


def smooth(a, k=5):
    n = len(a)
    r = [0.0] * n
    for i in range(n):
        s = 0
        cnt = 0
        for j in range(max(0, i - k // 2), min(n, i + k // 2 + 1)):
            s += a[j]
            cnt += 1
        r[i] = s / cnt
    return r


def runs_below(arr, thr):
    res = []
    s = None
    for i, v in enumerate(arr):
        if v < thr:
            if s is None:
                s = i
        elif s is not None:
            res.append((s, i))
            s = None
    if s is not None:
        res.append((s, len(arr)))
    return res


def find_inter_gaps(arr, n_expected, min_w=15,
                    thr_list=(0.15, 0.18, 0.20, 0.13, 0.22),
                    band=(0.06, 0.94)):
    """Find exactly n_expected interior black gaps between cells.

    Tries several thresholds (some sheets have JPEG ringing in gaps);
    rejects runs touching image edges (margins, may contain artifacts)
    and anything outside the central band; validates regular pitch.
    """
    a = smooth(arr, 5)
    mx = max(a)
    n = len(a)
    lo, hi = int(n * band[0]), int(n * band[1])
    for thr_f in thr_list:
        thr = mx * thr_f
        cand = [(s, e) for (s, e) in runs_below(a, thr)
                if s > 0 and e < n and e - s >= min_w
                and lo < (s + e) / 2 < hi]
        if len(cand) == n_expected:
            centers = [(s + e) / 2 for s, e in cand]
            pitches = [centers[i + 1] - centers[i]
                         for i in range(len(centers) - 1)]
            mean = sum(pitches) / len(pitches)
            var = sum((p - mean) ** 2 for p in pitches) / len(pitches)
            if var ** 0.5 < 4.0 and 170 < mean < 190:
                return cand, thr_f
    return None, None


def grid_for(path):
    """Return ((col_cells, row_cells, ...)) or (None, info) on failure."""
    col, row, w, h = profiles(path)
    cg, tc = find_inter_gaps(col, 9)
    rg, tr = find_inter_gaps(row, 4)
    if cg is None or rg is None:
        return None, (tc, tr)

    def cells(gaps):
        inner = [(gaps[i][1], gaps[i + 1][0]) for i in range(len(gaps) - 1)]
        widths = sorted(e - s for s, e in inner)
        med = widths[len(widths) // 2]
        first = (gaps[0][0] - med, gaps[0][0])
        last = (gaps[-1][1], gaps[-1][1] + med)
        return [first] + inner + [last], med

    cc, mw = cells(cg)
    rc, mh = cells(rg)
    return (cc, rc, mw, mh, tc, tr), None


def cut_sheet(part_path, out_paths):
    g, info = grid_for(part_path)
    if g is None:
        raise RuntimeError(f"grid detect failed for {part_path}: {info}")
    cc, rc, mw, mh, tc, tr = g
    im = Image.open(part_path).convert("RGB")
    w, h = im.size
    idx = 0
    for ry in range(5):
        for cx in range(10):
            x0, x1 = cc[cx]
            y0, y1 = rc[ry]
            cxp = int(round((x0 + x1) / 2))
            cyp = int(round((y0 + y1) / 2))
            hh = SIZE // 2
            box = (max(0, cxp - hh), max(0, cyp - hh),
                   min(w, cxp - hh + SIZE), min(h, cyp - hh + SIZE))
            crop = im.crop(box)
            if crop.size != (SIZE, SIZE):
                canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
                canvas.paste(crop, (0, 0))
                crop = canvas
            crop.save(out_paths[idx], optimize=True)
            idx += 1


def process_folder(folder):
    parts = sorted(glob.glob(os.path.join(folder, "*Part ?.jpg")))
    assert len(parts) == 2, f"expected 2 parts in {folder}, got {parts}"
    outdir = os.path.join(folder, "icons")
    os.makedirs(outdir, exist_ok=True)
    outs1 = [os.path.join(outdir, f"icon_{i:03d}.png") for i in range(1, 51)]
    outs2 = [os.path.join(outdir, f"icon_{i:03d}.png") for i in range(51, 101)]
    cut_sheet(parts[0], outs1)
    cut_sheet(parts[1], outs2)
    return outs1 + outs2


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    folders = sorted(glob.glob(os.path.join(root, "RPG Loot Icons *")))
    total = 0
    for f in folders:
        if only and only not in f:
            continue
        outs = process_folder(f)
        total += len(outs)
        print(f"OK {os.path.basename(f)}: {len(outs)} icons")
    print(f"TOTAL {total} icons")


if __name__ == "__main__":
    main()
