# Maui63 Data Archiving
### *[maui63.org](https://www.maui63.org/)*

_____

A basic repository for data archiving scripts and APIs for Maui63.  
🚧 This repository is still early stages and is a work in progress 🚧

## Installation

Clone the repository and install in your python environment using `pip install .`

### System Dependencies

The library relies on ffmepg and libjxl tools: `sudo apt install ffmpeg libjxl-tools`

## Usage

```
from maui63_data_archiving.converter import convert_images

input_dir = "path/to/images/"

# Convert to H265
convert_images(input_dir,"output.mkv", converter_type="h265", preset="veryslow")

# Convert to JXL
convert_images(input_dir,"jxl_output_dir", converter_type="jxl", quality=100)

```

## TODOs

- Transfer metadata
- Store frame names for video outputs


