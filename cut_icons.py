#!/usr/bin/env python3
"""
Cut RPG Loot Icon sheets into individual 149x149 icons.

Each sheet is a grid of 10 columns x 5 rows of icons. Every icon has a
gray-white outline (stroke) around it; the stroke + icon occupy a ~153x153
cell. We cut the INNER part of the outline (149x149), anchoring the crop box
at the TOP-RIGHT INNER corner of the stroke:

    crop = image[ top+2 : top+2+149 , right-2-148 : right-2 ]

where `top` is the topmost outline row of the icon and `right` the rightmost
outline column.

Grid detection:
  * Rows:      content runs of the full-height row profile (5 rows). If the
               irregular "RPG Loot Icons 01" sheets collapse two rows into
               one (their inter-row gaps are ~7px and carry faint
               antialiasing), retry with a higher brightness threshold.
  * Columns:   content runs PER ROW (10 cells), so sheets whose columns are
               not perfectly aligned between rows (set 01) still get their
               right edge right -- which is our anchor.

Per-icon refinement:
  * Each icon's own top outline row and right outline column are located by
    scanning for the nearest full-width / full-height light run (the stroke is
    a full line; icon art is not). When an icon's stroke is ambiguous we keep
    the grid-derived position (the "orient by previous-row distances" idea).

Output: RPG Loot Icons NN/setNN/partP_rowRR_colCC.png (100 icons per set) + icons_manifest.csv
"""
from PIL import Image
import numpy as np
import glob, re, os, csv

ICON = 149
MIN_RUN = 90      # cells are ~151-157 px; gaps are <= ~28 px
STROKE_RUN = 100  # a full stroke line is ~151-157 px

def label(path):
    m = re.search(r'RPG Loot Icons (\d+)/.*Part (\d+)\.jpg', path.replace('\\', '/'))
    return int(m.group(1)), int(m.group(2))

def runs_of(idx):
    if len(idx) == 0:
        return []
    out = []
    s = p = idx[0]
    for x in idx[1:]:
        if x == p + 1:
            p = x
        else:
            out.append((int(s), int(p)))
            s = p = x
    out.append((int(s), int(p)))
    return out

def detect_runs(prof, min_len, keep_n):
    idx = np.where(prof > 0)[0]
    runs = [r for r in runs_of(idx) if r[1] - r[0] + 1 >= min_len]
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    runs = runs[:keep_n]
    return sorted(runs)

def longest_run(mask1d):
    best = cur = 0
    for v in mask1d:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

def detect_rows(gray):
    for t in (30, 40, 50, 60, 70):
        rows = detect_runs((gray > t).sum(axis=1), MIN_RUN, 5)
        if len(rows) == 5:
            return rows
    return None

def detect_columns_in_band(band):
    for t in (30, 40, 50):
        cols = detect_runs((band > t).sum(axis=0), MIN_RUN, 10)
        if len(cols) == 10:
            return cols
    return None

def refine_top(gray, left, right, top):
    """Topmost row in [top-4, top+8] whose light run across the cell is full-width."""
    W, H = gray.shape[1], gray.shape[0]
    x0 = max(0, left - 2); x1 = min(W, right + 3)
    for y in range(max(0, top - 4), min(H, top + 9)):
        if longest_run(gray[y, x0:x1] > 30) >= STROKE_RUN:
            return y
    return top

def refine_right(gray, top, bottom, right):
    """Rightmost column in [right-8, right+4] whose light run down the cell is full-height."""
    W, H = gray.shape[1], gray.shape[0]
    y0 = max(0, top - 2); y1 = min(H, bottom + 3)
    for x in range(min(W - 1, right + 4), max(0, right - 9), -1):
        if longest_run(gray[y0:y1, x] > 30) >= STROKE_RUN:
            return x
    return right

def main():
    files = sorted(glob.glob('RPG Loot Icons */*.jpg'))
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = []
    problems = []

    for f in files:
        s, p = label(f)
        rgb = np.array(Image.open(f).convert('RGB'))
        gray = rgb.mean(axis=2)

        rows = detect_rows(gray)
        if rows is None:
            problems.append((s, p, 'rows', 'no 5 rows found'))
            continue

        for r, (top, bottom) in enumerate(rows):
            cols = detect_columns_in_band(gray[top:bottom])
            if cols is None:
                problems.append((s, p, f'row{r}', 'no 10 cols found'))
                continue
            for c, (left, right) in enumerate(cols):
                top2 = refine_top(gray, left, right, top)
                right2 = refine_right(gray, top, bottom, right)
                x0 = right2 - 2 - (ICON - 1)   # right - 150
                y0 = top2 + 2
                x1 = x0 + ICON
                y1 = y0 + ICON
                if x0 < 0 or y0 < 0 or x1 > rgb.shape[1] or y1 > rgb.shape[0]:
                    problems.append((s, p, f'r{r}c{c}', 'crop out of bounds'))
                    continue
                crop = rgb[y0:y1, x0:x1]
                out_dir = os.path.join(here, f'RPG Loot Icons {s:02d}', f'set{s:02d}')
                os.makedirs(out_dir, exist_ok=True)
                out = os.path.join(out_dir, f'part{p}_row{r:02d}_col{c:02d}.png')
                Image.fromarray(crop).convert('P', palette=Image.ADAPTIVE, colors=256).save(out, optimize=True)
                manifest.append([s, p, r, c, x0, y0, x1, y1, left, right, top, bottom,
                                 int(top2), int(right2)])

    with open(os.path.join(here, 'icons_manifest.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['set', 'part', 'row', 'col', 'x0', 'y0', 'x1', 'y1',
                    'cell_left', 'cell_right', 'cell_top', 'cell_bottom',
                    'refined_top', 'refined_right'])
        w.writerows(manifest)

    print(f"Total icons cut: {len(manifest)}")
    print(f"Problems: {len(problems)}")
    for pr in problems:
        print("  ", pr)

    total = 0
    for setdir in glob.glob(os.path.join(here, 'RPG Loot Icons *', 'set*')):
        for fn in os.listdir(setdir):
            if fn.endswith('.png'):
                total += os.path.getsize(os.path.join(setdir, fn))
    print(f"Total PNG bytes: {total/1e6:.1f} MB")

if __name__ == '__main__':
    main()
