#!/usr/bin/env python3
"""
Уменьшение RPG Loot Icons 01 (147px) до 128x128 с подавлением шума
при сохранении деталей.

Пайплайн (подобран по метрикам шума/деталей и визуальному сравнению):
1. Lanczos4-ресайз 147 -> 128 (максимальное сохранение мелких деталей).
2. Нелокальное шумоподавление (fastNlMeansDenoisingColored, h=3) уже на
   128px: JPEG-шум тёмного фона убирается точечно, края и микроконтраст
   иконок не размываются.
3. Сохранение в PNG без потерь.

Подавление после ресайза выигрывает у подавления до ресайза: шум мельчает
и локально становится более «плоским», поэтому h=3 достаточно, а детали
не страдают. Unsharp-маска после шумодава отклонена: +1-3% резкости ценой
+5-14% шума и ореолов.
"""
import os
import glob
import cv2

SRC_DIR = "RPG Loot Icons 01"
DST_DIR = "RPG Loot Icons 01_128_v2"
SIZE = 128
NLM_H = 3          # сила шумоподавления (малая => детали в приоритете)


def process(src_path, dst_path):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    img128 = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)
    clean = cv2.fastNlMeansDenoisingColored(img128, None, NLM_H, NLM_H, 7, 21)
    cv2.imwrite(dst_path, clean)


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC_DIR, "icon_*.png")))
    assert len(files) == 100, f"ожидалось 100 исходников, найдено {len(files)}"
    for f in files:
        process(f, os.path.join(DST_DIR, os.path.basename(f)))
    print(f"Обработано {len(files)} иконок -> {DST_DIR}/")


if __name__ == "__main__":
    main()
