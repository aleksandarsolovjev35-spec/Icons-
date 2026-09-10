#!/usr/bin/env python3
"""Rebuild the approved polished 256×256 RPG loot icons from source PNGs.

The script is intentionally conservative: it reads originals, writes only to a
separate output directory, and refuses to overwrite matching output PNGs unless
--overwrite is explicitly supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


FINAL_SIZE = (256, 256)
INTERMEDIATE_SIZE = (512, 512)
PNG_COMPRESSION = 9


def process_icon(image: np.ndarray) -> np.ndarray:
    """Apply the exact approved detail-preserving, polished pipeline."""
    # Mild native-resolution cleanup preserves painted marks and fine texture.
    cleaned = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=5,
        hColor=6,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # Polished 512 px intermediate followed by anti-aliased 256 px output.
    intermediate = cv2.resize(
        cleaned, INTERMEDIATE_SIZE, interpolation=cv2.INTER_CUBIC
    )
    polished = cv2.resize(
        intermediate, FINAL_SIZE, interpolation=cv2.INTER_AREA
    )

    # Gentle, thresholded contour contrast in lightness only—not a sharpen pass.
    lab = cv2.cvtColor(polished, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    local_lightness = cv2.GaussianBlur(
        lightness, (0, 0), sigmaX=0.85, sigmaY=0.85
    )
    edge_detail = lightness - local_lightness
    edge_mask = np.clip((np.abs(edge_detail) - 7.0) / 16.0, 0.0, 1.0)
    lab[:, :, 0] = np.clip(
        lightness + 0.08 * edge_detail * edge_mask, 0.0, 255.0
    )

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild polished 256×256 RPG loot icons from source PNG files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing the original icon_*.png files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Separate directory to receive the 256×256 PNG files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of output PNG files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Input directory does not exist: {source}")
    if source == destination:
        raise SystemExit("--input and --output must be different directories.")

    files = sorted(source.glob("*.png"))
    if not files:
        raise SystemExit(f"No PNG files found in: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    existing = [destination / path.name for path in files if (destination / path.name).exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite {len(existing)} existing PNG file(s) in {destination}. "
            "Choose an empty --output directory or add --overwrite."
        )

    written = 0
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not decode: {path}")

        result = process_icon(image)
        if result.shape != (FINAL_SIZE[1], FINAL_SIZE[0], 3):
            raise RuntimeError(f"Unexpected output shape for {path.name}: {result.shape}")

        target = destination / path.name
        succeeded = cv2.imwrite(
            str(target), result, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
        )
        if not succeeded:
            raise SystemExit(f"Could not write: {target}")
        written += 1

    print(f"Created {written} RGB PNG icon(s) at {FINAL_SIZE[0]}×{FINAL_SIZE[1]} in {destination}")


if __name__ == "__main__":
    main()
