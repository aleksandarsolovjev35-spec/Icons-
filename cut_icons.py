#!/usr/bin/env python3
"""Cut RPG Loot Icons sheets into individual icons.

Each "RPG Loot Icons NN" folder contains 2 sheets ("Part 1/2", 1800x1013 JPG,
10 cols x 5 rows = 50 icons each). This script detects the grid of every
sheet automatically (grid position varies slightly between sheets; folders
39-40 have no cell borders) and saves 100 PNG icons per folder:

    RPG Loot Icons NN/icons/icon_001.png ... icon_100.png
    (Part 1 -> 001-050, Part 2 -> 051-100, row-major order)

Icons are 144x144, cropped inside each cell's real gray border lines
(detected by snapping; 2px margin off the lines), on the original black
background. Borderless sheets (39-40) are cropped centered on each cell.

Usage:
    python3 cut_icons.py            # process all folders
    python3 cut_icons.py 01         # process only folders matching "01"

Requires: Pillow (pip install pillow). No numpy/cv2 needed.
"""
import glob
import os
import sys

from PIL import Image

SIZE = 144  # uniform output icon size (px), centered on detected cell
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


def _snap_edge(px, w, h, x0, x1, y0, y1):
    """Snap a cell box to its real gray border lines.

    Gap-profile detection can be off by a few px when a bright icon touches
    the border (JPEG ringing). The border itself is unambiguous: a full-height
    (resp. full-width) line of gray pixels. Returns refined (bl, bt, br, bb)
    border lines, or None if not found (e.g. borderless sheets).
    """
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)

    def col_score(xc):
        n = t = 0
        for y in range(y0 + 12, y1 - 12, 2):
            t += 1
            if 40 <= px[xc, y] <= 130:
                n += 1
        return n / t

    def row_score(yc):
        n = t = 0
        for x in range(x0 + 12, x1 - 12, 2):
            t += 1
            if 40 <= px[x, yc] <= 130:
                n += 1
        return n / t

    def pick(cands, ref, inner_sign):
        # nearest candidate to ref; tie-break towards cell interior
        best = None
        for c, s in cands:
            key = (abs(c - ref), -inner_sign * c)
            if best is None or key < best[0]:
                best = (key, c)
        return best[1] if best else None

    bl = pick([(x, col_score(x)) for x in range(max(0, x0 - 8), x0 + 13)
               if col_score(x) > 0.7], x0, +1)
    br = pick([(x, col_score(x)) for x in range(x1 - 12, min(w, x1 + 9))
               if col_score(x) > 0.7], x1, -1)
    bt = pick([(y, row_score(y)) for y in range(max(0, y0 - 8), y0 + 13)
               if row_score(y) > 0.7], y0, +1)
    bb = pick([(y, row_score(y)) for y in range(y1 - 12, min(h, y1 + 9))
               if row_score(y) > 0.7], y1, -1)
    # per-edge fallback to the profile-detected box (may be a few px off when
    # an icon touches the border, or on borderless sheets): keeps every edge
    # as accurate as possible instead of discarding all four.
    if bl is None:
        bl = x0
    if br is None:
        br = x1
    if bt is None:
        bt = y0
    if bb is None:
        bb = y1
    if not (135 <= br - bl <= 162 and 135 <= bb - bt <= 162):
        return None
    return bl, bt, br, bb


def cut_sheet(part_path, out_paths):
    g, info = grid_for(part_path)
    if g is None:
        raise RuntimeError(f"grid detect failed for {part_path}: {info}")
    cc, rc, mw, mh, tc, tr = g
    im = Image.open(part_path).convert("RGB")
    gr = Image.open(part_path).convert("L")
    px = gr.load()
    w, h = im.size
    idx = 0
    for ry in range(5):
        for cx in range(10):
            x0, x1 = cc[cx]
            y0, y1 = rc[ry]
            snap = _snap_edge(px, w, h, x0, x1, y0, y1)
            if snap is not None:
                # interior between real borders, 2px margin off the lines
                bl, bt, br, bb = snap
                ix0, iy0, ix1, iy1 = bl + 2, bt + 2, br - 2, bb - 2
                # uniform SIZE crop centered in interior (fallback: interior)
                cxp = (ix0 + ix1) / 2
                cyp = (iy0 + iy1) / 2
                hh = SIZE // 2
                if ix1 - ix0 >= SIZE and iy1 - iy0 >= SIZE:
                    box = (int(round(cxp - hh)), int(round(cyp - hh)),
                           int(round(cxp - hh)) + SIZE,
                           int(round(cyp - hh)) + SIZE)
                else:
                    box = (int(ix0), int(iy0), int(ix1), int(iy1))
            else:
                # borderless sheet: centered crop on detected cell
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
