import os
import tempfile
from pathlib import Path
from typing import List, Union

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
        return iio3.get_frame_count(self.video_path)
    
    def get_image(self, idx) -> np.ndarray:
        return iio3.imread(self.video_path, index=idx)
    
    def get_image_path(self, idx):
        # Create temp files for frames
        image = Image.fromarray(self.get_image(idx))
        image.save(os.path.join(self.tmp_dir.name, idx))
