"""Generate synthetic assets and run an end-to-end mosaic retrieval demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from mosaic_pipeline import (
    extract_features_from_folder,
    extract_tile_features,
    match_tiles_to_gallery_faiss,
    reconstruct_mosaic_image,
)


TILE_SIZE = 32
COLORS = {
    "red": (210, 55, 65),
    "green": (55, 165, 105),
    "blue": (55, 105, 205),
    "gold": (225, 175, 55),
}


def create_demo_assets(output_dir: Path) -> tuple[Path, Path]:
    """Create a geometric target and a small synthetic gallery."""

    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    for index, (name, color) in enumerate(COLORS.items()):
        image = Image.new("RGB", (TILE_SIZE, TILE_SIZE), color)
        draw = ImageDraw.Draw(image)
        bounds = (index * 3, index * 3, 31 - index * 3, 31 - index * 3)
        draw.rectangle(bounds, outline="white", width=2)
        image.save(gallery_dir / f"{name}.png")

    layout = [
        ["red", "red", "green", "green"],
        ["red", "gold", "gold", "green"],
        ["blue", "gold", "gold", "red"],
        ["blue", "blue", "green", "red"],
    ]
    target = Image.new("RGB", (TILE_SIZE * 4, TILE_SIZE * 4))
    for row_index, row in enumerate(layout):
        for column_index, color_name in enumerate(row):
            tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), COLORS[color_name])
            target.paste(tile, (column_index * TILE_SIZE, row_index * TILE_SIZE))

    target_path = output_dir / "target.png"
    target.save(target_path)
    return target_path, gallery_dir


def run_demo(output_dir: Path) -> Path:
    """Run the full RGB-feature retrieval pipeline and return the mosaic path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path, gallery_dir = create_demo_assets(output_dir)
    gallery_features = output_dir / "gallery_features.pkl"
    tile_features = output_dir / "tile_features.pkl"
    matches = output_dir / "matches.pkl"
    mosaic = output_dir / "mosaic.png"

    extract_features_from_folder(str(gallery_dir), "avg_rgb", str(gallery_features))
    extract_tile_features(str(target_path), TILE_SIZE, "avg_rgb", str(tile_features))
    match_tiles_to_gallery_faiss(str(tile_features), str(gallery_features), str(matches))
    reconstruct_mosaic_image(str(matches), str(gallery_dir), TILE_SIZE, str(mosaic))
    return mosaic


def main() -> int:
    """Parse arguments and run the synthetic demo."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    mosaic = run_demo(args.output_dir)
    print(f"Demo mosaic written to {mosaic.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
