#!/usr/bin/env python3
"""
Нарезка RPG-иконок из спрайтшитов (сетка 10 колонок x 5 рядов).

Алгоритм:
1. Для каждого столбца/строки вычисляется средняя яркость; непрерывные
   участки с яркостью < 20 и шириной >= 20 px — разделительные полосы.
2. Ячейки — прямоугольники между соседними разделителями. Если крайнее
   поле слишком узкое и не обнаружено как полоса, оно восстанавливается
   по регулярному шагу сетки (виртуальная полоса).
3. Серая рамка ячейки обрезается: границы сдвигаются на 3 px внутрь
   (153 - 2*3 = 147 px).
4. Каждая иконка сохраняется в PNG (без потерь) в папку назначения.
"""
import os
import numpy as np
from PIL import Image

SOURCES = [
    ("file (3).jpg", 1),   # иконки 001..050
    ("file (4).jpg", 51),  # иконки 051..100
]
OUT_DIR = "RPG Loot Icons 02"
DARK_THRESH = 20   # яркость разделителя
MIN_BAND_W = 20    # минимальная ширина полосы, px
FRAME_TRIM = 3     # отступ внутрь ячейки для среза серой рамки, px
N_COLS, N_ROWS = 10, 5


def find_bands(brightness, min_width=MIN_BAND_W, thresh=DARK_THRESH):
    """Непрерывные участки с яркостью < thresh и шириной >= min_width."""
    dark = brightness < thresh
    bands, start = [], None
    for i, d in enumerate(dark):
        if d and start is None:
            start = i
        elif not d and start is not None:
            if i - start >= min_width:
                bands.append([start, i])  # [start, end)
            start = None
    if start is not None and len(dark) - start >= min_width:
        bands.append([start, len(dark)])
    return bands


def add_virtual_edge_bands(bands, total):
    """Если крайнее поле листа уже 20 px, оно не обнаружено как полоса.

    Восстанавливаем его по шагу сетки, чтобы каждая ячейка оказалась
    зазором между двумя соседними полосами.
    """
    bands = [list(b) for b in bands]
    centers = [(a + b) / 2 for a, b in bands]
    k = np.arange(len(centers))
    pitch = float(np.polyfit(k, centers, 1)[0])
    band_w = float(np.median([b - a for a, b in bands]))

    if bands[0][0] > 0:  # слева полосы нет — добавляем виртуальную
        vs = bands[0][0] - pitch
        bands.insert(0, [int(round(vs)), int(round(vs + band_w))])
    if bands[-1][1] < total:  # справа полосы нет — добавляем виртуальную
        vs = bands[-1][1] + pitch - band_w
        bands.append([int(round(vs)), int(round(min(vs + band_w, total)))])
    return bands


def cell_bounds(bands, total, expected):
    """Ячейки — зазоры между соседними полосами разделителей."""
    bands = add_virtual_edge_bands(bands, total)
    cells = [(bands[i][1], bands[i + 1][0]) for i in range(len(bands) - 1)]

    widths = [e - s for s, e in cells]
    if len(cells) != expected or max(abs(w - np.median(widths)) for w in widths) > 5:
        raise ValueError(f"Ожидалось {expected} ячеек ~{np.median(widths)}px, "
                         f"получено {len(cells)}: {list(zip(cells, widths))}")
    return cells


def extract_sheet(path, out_dir, first_index):
    img = Image.open(path).convert("RGB")
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    H, W = gray.shape

    col_cells = cell_bounds(find_bands(gray.mean(axis=0)), W, N_COLS)
    row_cells = cell_bounds(find_bands(gray.mean(axis=1)), H, N_ROWS)
    # единый итоговый размер: ширина ячейки минус рамка с двух сторон
    cell_w = int(round(np.median([e - s for s, e in col_cells])))
    cell_h = int(round(np.median([e - s for s, e in row_cells])))
    out_w, out_h = cell_w - 2 * FRAME_TRIM, cell_h - 2 * FRAME_TRIM
    print(f"{path}: {len(col_cells)}x{len(row_cells)} ячеек, "
          f"ячейка {cell_w}x{cell_h} px -> иконка {out_w}x{out_h} px")
    print(f"  колонки: x={col_cells[0]} ... x={col_cells[-1]}")
    print(f"  ряды:    y={row_cells[0]} ... y={row_cells[-1]}")

    saved = []
    idx = first_index
    for (y0, y1) in row_cells:
        for (x0, x1) in col_cells:
            # рамка срезается на 3 px внутрь от края ячейки
            crop = img.crop((x0 + FRAME_TRIM, y0 + FRAME_TRIM,
                             x0 + FRAME_TRIM + out_w, y0 + FRAME_TRIM + out_h))
            name = f"icon_{idx:03d}.png"
            crop.save(os.path.join(out_dir, name), "PNG")  # PNG = без потерь
            saved.append((name, crop.size))
            idx += 1
    return saved


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = []
    for path, first in SOURCES:
        total += extract_sheet(path, OUT_DIR, first)

    sizes = {s for _, s in total}
    print(f"\nСохранено {len(total)} иконок в '{OUT_DIR}/', размеры: {sizes}")


if __name__ == "__main__":
    main()
