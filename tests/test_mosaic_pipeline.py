from pathlib import Path

import numpy as np
from PIL import Image

from mosaic_pipeline import normalize_rows, split_image_into_tiles


def test_normalize_rows_is_float32_and_zero_safe() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)

    result = normalize_rows(matrix)

    assert result.dtype == np.float32
    assert np.allclose(result[0], [0.6, 0.8])
    assert np.allclose(result[1], [0.0, 0.0])
    assert result.flags["C_CONTIGUOUS"]


def test_split_image_ignores_incomplete_border_tiles(tmp_path: Path) -> None:
    image_path = tmp_path / "target.png"
    Image.new("RGB", (70, 65), "navy").save(image_path)

    tiles = split_image_into_tiles(str(image_path), tile_size=32)

    assert len(tiles) == 4
    assert [tile_id for tile_id, _ in tiles] == [
        "target_tile_000_000",
        "target_tile_000_001",
        "target_tile_001_000",
        "target_tile_001_001",
    ]
