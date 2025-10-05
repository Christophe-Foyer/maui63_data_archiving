"""
Simple image conversion system supporting multiple input sources and output formats.
"""

from pathlib import Path
from typing import List, Union

import numpy as np
from maui63_data_archiving.converters.h265 import H265Converter
from maui63_data_archiving.converters.jxl import JPEGXLConverter
from maui63_data_archiving.image_source.folder import FolderSource, ImageListSource
from maui63_data_archiving.image_source.video import VideoSource
from tqdm.auto import tqdm


def convert_images(
    source: Union[List[Union[str, Path]], str, Path],
    output_path: Union[str, Path],
    converter_type: str = "h265",
    check_frames=False,
    **kwargs
):
    """
    Main conversion function.
    
    Args:
        source: Can be:
            - List of image paths
            - Path to folder containing images
            - Path to video file
        output_path: Output location (file for video, folder for images)
        converter_type: Type of converter ("h265" or "jxl")
        **kwargs: Additional parameters passed to converter
        
    Example:
        # Convert folder to H.265 video
        convert_images("./images", "output.mkv", converter_type="h265", preset="fast")
        
        # Convert video to JPEG-XL images
        convert_images("video.mp4", "./output_jxl", converter_type="jxl", quality=95)
        
        # Convert image list to video
        convert_images(["img1.png", "img2.png"], "output.mkv", converter_type="h265")
    """
    # Determine source type
    if isinstance(source, list):
        img_source = ImageListSource(source)
    elif isinstance(source, (str, Path)):
        source = Path(source)
        if source.is_dir():
            pattern = kwargs.pop("pattern", "*.png")
            img_source = FolderSource(source, pattern=pattern)
        elif source.is_file():
            img_source = VideoSource(source)
        else:
            raise ValueError(f"Source path does not exist: {source}")
    else:
        raise ValueError(f"Invalid source type: {type(source)}")
    
    # Select converter
    converters = {
        "h265": H265Converter(),
        "jxl": JPEGXLConverter(),
    }
    
    if converter_type not in converters:
        raise ValueError(f"Converter type must be one of {list(converters.keys())}")
    
    converter = converters[converter_type]
    
    # Convert
    print(f"Converting {len(img_source)} images using {converter_type}...")
    converter.convert(img_source, output_path, **kwargs)
    print("Conversion complete!")

    # Verify if requested
    if check_frames:
        output = Path(output_path)
        if output.is_dir():
            # TODO: This will read all files, should we somehow get an output file list?
            img_output = FolderSource(output, pattern="*")
        elif output.is_file():
            img_output = VideoSource(output)
        else:
            raise ValueError(f"Output path does not exist: {output}")

        print("Verifying lossless conversion...")
        for idx in tqdm(range(len(img_source)), desc="Checking frames"):
            original = img_source.get_image(idx)
            readback = img_output.get_image(idx)
            np.testing.assert_array_equal(original, readback)
        print("Verification passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert images or videos between different formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert folder to H.265 video
  python script.py ./images output.mkv --converter h265 --preset fast --fps 30
  
  # Convert folder to JPEG-XL images
  python script.py ./images ./output_jxl --converter jxl --quality 95
  
  # Convert video to H.265 with CRF
  python script.py input.mp4 output.mkv --converter h265 --crf 20 --no-lossless
  
  # Convert with lossless verification
  python script.py ./images output.mkv --converter h265 --check-frames
        """
    )
    
    # Required arguments
    parser.add_argument("source", help="Source: folder path, video file, or image files")
    parser.add_argument("output", help="Output path (file for video, folder for images)")
    
    # Converter selection
    parser.add_argument(
        "--converter", "-c",
        choices=["h265", "jxl"],
        default="h265",
        help="Conversion format (default: h265)"
    )
    parser.add_argument(
        "--check-frames",
        action="store_true",
        help="Verify lossless conversion (only works with lossless formats!)"
    )
    
    # H.265 specific options
    h265_group = parser.add_argument_group("H.265 options")
    h265_group.add_argument(
        "--preset", "-p",
        choices=H265Converter.PRESETS,
        default="medium",
        help="x265 encoding preset (default: medium)"
    )
    h265_group.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video output (default: 30)"
    )
    # h265_group.add_argument(
    #     "--crf",
    #     type=int,
    #     choices=range(0, 52),
    #     metavar="0-51",
    #     help="Constant Rate Factor for lossy encoding (lower=better, typical: 18-28)"
    # )
    # h265_group.add_argument(
    #     "--lossless",
    #     action="store_true",
    #     help="Use lossless encoding"
    # )
    
    # JPEG-XL specific options
    jxl_group = parser.add_argument_group("JPEG-XL options")
    jxl_group.add_argument(
        "--quality", "-q",
        type=int,
        default=100,
        choices=range(0, 101),
        metavar="0-100",
        help="JPEG-XL quality (100=lossless, default: 100)"
    )
    
    # Image source options
    source_group = parser.add_argument_group("Image source options")
    source_group.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern for folder source (default: *.png)"
    )
    
    args = parser.parse_args()
    
    # Build kwargs based on converter type
    kwargs = {}
    
    if args.converter == "h265":
        kwargs.update({
            "preset": args.preset,
            "fps": args.fps,
            # "lossless": args.lossless,
        })
        # if args.crf is not None:
        #     kwargs["crf"] = args.crf
    elif args.converter == "jxl":
        kwargs["quality"] = args.quality
    
    # Add pattern for folder sources
    kwargs["pattern"] = args.pattern
    
    # Run conversion
    try:
        convert_images(
            args.source,
            args.output,
            converter_type=args.converter,
            check_frames=args.check_frames,
            **kwargs
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
