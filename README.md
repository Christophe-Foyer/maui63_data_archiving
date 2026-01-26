# Maui63 Data Archiving
### *[maui63.org](https://www.maui63.org/)*

_____

A basic repository for data archiving scripts and APIs for Maui63.  
🚧 This repository is still early stages and is a work in progress 🚧

## Installation

This project uses `uv` for dependency management.

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    ```bash
    uv sync --all-extras
    ```
    This will create a virtual environment in `.venv` with all necessary dependencies, including ML libraries.

### System Dependencies

The library relies on `ffmpeg` and `libjxl-tools`:
```bash
sudo apt install ffmpeg libjxl-tools
```

## Usage

## Usage

### Basic Conversion
Run scripts using `uv run` to ensure dependencies from the environment are used.

```python
from maui63_data_archiving.converter import convert_images

input_dir = "path/to/images/"

# Convert to H265
convert_images(input_dir, "output.mkv", converter_type="h265", preset="veryslow")

# Convert to JXL
convert_images(input_dir, "jxl_output_dir", converter_type="jxl", quality=100)
```

### Machine Learning
Train the classifier using the provided script:

```bash
uv run python maui63_data_archiving/ml/train.py
```

This script expects data in `data/external` by default.

### Data
The project uses COCO-formatted datasets.
To download sample training data, you can refer to the links below or use your own dataset structure:
`data/external/{dataset_name}/{train,valid,test}/_annotations.coco.json`

## TODOs

- Transfer metadata
- Store frame names for video outputs
- Frame Culling based on anomaly detection (WIP)

# Frame Culling Dataset

Currently need to coordinate getting actual data, but I'm playing around with some license-free topdown ocean footage.

```
Training data:
https://pixabay.com/videos/waves-ocean-beach-turquoise-sea-65562/
https://pixabay.com/videos/sea-ocean-wave-beach-blue-nature-24216/
https://pixabay.com/videos/beach-sea-ocean-sand-drone-aerial-186115/
https://pixabay.com/videos/drone-nature-landscape-air-photo-23334/
https://pixabay.com/videos/ocean-beach-waves-sea-sand-nature-205223/
https://pixabay.com/videos/beach-shore-aerial-view-reef-sea-7262/
https://pixabay.com/videos/nature-ocean-sea-shore-coastline-35263/
https://pixabay.com/videos/ocean-sea-waves-water-beach-rocks-231738/
https://pixabay.com/videos/waves-ocean-sea-beach-rocks-water-231739/
```

```
Test data:
https://pixabay.com/videos/birds-travel-sea-ocean-beach-203315/
https://pixabay.com/videos/drone-flying-camera-4k-drone-277315/
```