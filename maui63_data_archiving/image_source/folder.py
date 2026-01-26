from pathlib import Path
from typing import List, Union

import imageio.v3 as iio3
import numpy as np
import pillow_jxl

_ = pillow_jxl  # So stuff doesn't complain, I'm sure there's an ignore flag in vscode somewhere though

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from maui63_data_archiving.image_source.abstract import ImageSource


class ImageListSource(ImageSource):
    """Source from a list of image paths."""

    def __init__(self, image_paths: List[Union[str, Path]]):
        self.image_paths = [Path(p) for p in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def get_image(self, idx) -> np.ndarray:
        return iio3.imread(self.get_image_path(idx), plugin="pillow")

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

    def get_image(self, idx) -> np.ndarray:
        return iio3.imread(self.get_image_path(idx), plugin="pillow")

    def get_image_path(self, idx) -> str:
        return self.image_paths[idx]
