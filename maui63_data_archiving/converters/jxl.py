import subprocess
from pathlib import Path
from typing import Union

import pillow_jxl as _
from tqdm.autonotebook import tqdm

from maui63_data_archiving.converters.abstract import Converter
from maui63_data_archiving.image_source.abstract import ImageSource


class JPEGXLConverter(Converter):
    """Convert images to JPEG-XL format."""
    
    def convert(
        self,
        source: ImageSource,
        output_path: Union[str, Path],
        quality: int = 100,  # 100% = lossless
        output_suffix: str = ".jxl",
        **kwargs
    ):
        """
        Convert images to JPEG-XL format.
        
        Args:
            source: Image source
            output_path: Output folder path for images
            quality: Quality setting (0-100, 100 = lossless)
            output_suffix: File extension for output files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # TODO: Parallelize this
        for idx in tqdm(range(len(source)), total=len(source), desc="Converting to JPEG-XL"):
            img_path = source.get_image_path(idx)

            output_file = output_path / (img_path.stem + output_suffix)
            
            # TODO: use pillow-jxl-plugin to write instead?
            cmd = ["cjxl", str(img_path), str(output_file), "-q", str(quality)]
            subprocess.run(cmd, check=True, capture_output=True)
        
        print(f"Converted {len(source)} images to {output_path}")
