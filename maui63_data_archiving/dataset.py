from torch.utils.data import Dataset, DataLoader
from maui63_data_archiving.image_source.video import VideoSource
from maui63_data_archiving.image_source.folder import FolderSource
from lightning.pytorch import LightningDataModule
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
import warnings
import numpy as np
from pathlib import Path


# TODO: There's no reason why this can only handle video sources, could also handle folder sources
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

# TODO: Integrate a labeling pipeline where we can include masks
# Should also default to an empty mask so we don't have to label most images, just those with "anomalies"
class FrameDataset(Dataset):
    def __init__(self, data_path, tile_size: int=None, transform=None):
        source = Path(data_path)
        if source.is_dir():
            self.source = FolderSource(source, pattern="*")
        elif source.is_file():
            # TODO: Preloading/caching frames would probably help a lot on the video data
            self.source = VideoSource(source)
        else:
            raise FileExistsError(source)

        self.framecount = len(self.source)
        self.transform = transform

        # Assumes the image shape doesn't change in the source (could be an issue with folders?)
        # TODO: Move this to a subfunction so we can call it on each getitem (probably with caching)
        im = self.source.get_image(0)
        self.img_h, self.img_w = im.shape[:2]

        self.tile_size = tile_size
        if tile_size:
            self.tiles_x = int(np.ceil(self.img_w / tile_size))
            self.tiles_y = int(np.ceil(self.img_h / tile_size))
            self.tiles_per_frame = self.tiles_x * self.tiles_y

            # Compute overlap strides
            self.stride_x = (self.img_w - tile_size) / max(self.tiles_x - 1, 1)
            self.stride_y = (self.img_h - tile_size) / max(self.tiles_y - 1, 1)
        else:
            self.tiles_x = self.tiles_y = 1
            self.tiles_per_frame = 1
            self.stride_x = self.stride_y = 0

    def __len__(self):
        return self.framecount * self.tiles_per_frame

    def __getitem__(self, idx):
        frame_idx = idx // self.tiles_per_frame
        tile_idx = idx % self.tiles_per_frame

        tile_y_idx = tile_idx // self.tiles_x
        tile_x_idx = tile_idx % self.tiles_x

        frame = self.source.get_image(frame_idx)

        if self.tile_size:
            x0 = int(tile_x_idx * self.stride_x)
            y0 = int(tile_y_idx * self.stride_y)
            x1 = x0 + self.tile_size
            y1 = y0 + self.tile_size

            # Clip just in case of floating-point rounding
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(self.img_w, x1), min(self.img_h, y1)

            tile = frame[y0:y1, x0:x1, :]
        else:
            tile = frame

        out = {"image": tile, "label": 0}

        if self.transform:
            out = self.transform(**out)

        return out
