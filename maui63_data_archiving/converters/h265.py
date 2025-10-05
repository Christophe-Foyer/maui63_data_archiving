import subprocess
from pathlib import Path
from typing import Union

import imageio.v2 as iio2
import imageio.v3 as iio3
import numpy as np
import skimage.io
from tqdm.autonotebook import tqdm

from maui63_data_archiving.image_source.abstract import ImageSource
from maui63_data_archiving.converters.abstract import Converter


class H265Converter(Converter):
    """Convert images to lossless H.265 video."""
    
    PRESETS = [
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow", "placebo"
    ]
    
    def convert(
        self,
        source: ImageSource,
        output_path: Union[str, Path],
        preset: str = "veryslow",
        fps: int = 1,
        **kwargs
    ):
        """
        Convert images to lossless H.265 video.
        
        Args:
            source: Image source
            output_path: Output video file path
            preset: x265 preset (speed/compression tradeoff)
            fps: Frames per second
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Preset must be one of {self.PRESETS}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(source, "video_path"): # Convert directly with ffmpeg for videos
            cmd = [
                "ffmpeg", "-y",
                "-i", str(source.video_path),
                "-c:v", "libx265",
                "-preset", preset,
                "-x265-params", "lossless=1",
                "-pix_fmt", "gbrp",
                str(output_path)
            ]

            print(f"Converting video directly: {source.video_path} → {output_path}")
            subprocess.run(cmd, check=True)
        else: # It's an image source
            # Ensure output path has proper extension
            if not output_path.suffix:
                output_path = output_path.with_suffix('.mp4')
            
            # Setup writer
            writer = iio2.get_writer(
                str(output_path),
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx265",
                output_params=["-preset", preset, "-x265-params", "lossless=1", "-pix_fmt", "gbrp"],
            )
            
            # Write frames directly from disk
            for idx in tqdm(range(len(source)), desc="Writing video"):
                image_path = source.get_image_path(idx)
                if image_path is None:
                    raise ValueError("Source does not support file-based access")
                img = skimage.io.imread(str(image_path))
                writer.append_data(img)
            
            writer.close()
