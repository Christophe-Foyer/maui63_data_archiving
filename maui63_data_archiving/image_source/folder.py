from pathlib import Path
from typing import List, Union
from functools import lru_cache

import cv2
import numpy as np
import pillow_jxl  # noqa # pylint: disable=unused-import
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from maui63_data_archiving.image_source.abstract import ImageSource


def robust_read_image(path: str) -> np.ndarray:
    """Read image with OpenCV, falling back to PIL for truncated/corrupted JPEGs."""
    # Try OpenCV first (fast)
    image = cv2.imread(path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Fallback to PIL (tolerant)
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception as e:
        raise IOError(
            f"Failed to load image even with PIL fallback: {path}. Error: {e}"
        )


class ImageListSource(ImageSource):
    """Source from a list of image paths."""

    def __init__(self, image_paths: List[Union[str, Path]]):
        self.image_paths = [Path(p) for p in image_paths]

    def __len__(self):
        return len(self.image_paths)

    @lru_cache(maxsize=2)
    def get_image(self, idx) -> np.ndarray:
        path = str(self.get_image_path(idx))
        return robust_read_image(path)

    def get_image_path(self, idx) -> str:
        return self.image_paths[idx]


class FolderSource(ImageSource):
    """Source from a folder containing images."""

    def __init__(self, folder_path: Union[str, Path], pattern: str = "*.png"):
        self.image_paths = sorted(folder_path.glob(pattern))

        if not self.image_paths:
            raise ValueError(f"No images matching '{pattern}' found in {folder_path}")

    def __len__(self):
        return len(self.image_paths)

    @lru_cache(maxsize=2)
    def get_image(self, idx) -> np.ndarray:
        path = str(self.get_image_path(idx))
        return robust_read_image(path)

    def get_image_path(self, idx) -> str:
        return self.image_paths[idx]
