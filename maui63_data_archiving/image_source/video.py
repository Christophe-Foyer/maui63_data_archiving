import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Union

import imageio
import imageio.v3 as iio3
import numpy as np
from PIL import Image

from maui63_data_archiving.image_source.abstract import ImageSource


class VideoSource(ImageSource):
    """Source from a video file."""

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.tmp_dir = tempfile.TemporaryDirectory(prefix=os.path.basename(video_path))

    def __len__(self):
        # Using v3 for frame count as well
        props = iio3.improps(self.video_path, plugin="ffmpeg")
        return (
            props.shape[0]
            if props.shape
            else imageio.get_reader(self.video_path, "ffmpeg").count_frames()
        )

    @lru_cache(maxsize=2)
    def get_image(self, idx) -> np.ndarray:
        return iio3.imread(self.video_path, index=idx, plugin="ffmpeg")

    def get_image_path(self, idx):
        # Create temp files for frames
        image = Image.fromarray(self.get_image(idx))
        image.save(os.path.join(self.tmp_dir.name, str(idx)))
