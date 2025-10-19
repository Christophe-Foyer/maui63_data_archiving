from torch.utils.data import Dataset, DataLoader
from maui63_data_archiving.image_source.video import VideoSource
from pytorch_lightning import LightningDataModule
import warnings
import numpy as np


# TODO: There's no reason why this can only handle video sources, could also handle folder sources
class VideoFrameDataset(Dataset):
    # TODO: Support tiling (wip)

    def __init__(self, video_path, tile_size: int=None, transform=None):
        # Assumes square tiles?

        self.source = VideoSource(video_path)
        self.framecount = len(self.source)
        self.transform = transform

        # Assumes the image shape doesn't change (can it even do that?) 
        im = self.source.get_image(0)
        self.img_h, self.img_w = im.shape[:2]

        self.tile_size = tile_size
        if tile_size:
            self.tiles_x = int(np.ceil(self.img_w / tile_size))
            self.tiles_y = int(np.ceil(self.img_h / tile_size))
            self.tiles_per_frame = self.tiles_x * self.tiles_y
        else:
            self.tiles_x = self.tiles_y = 1
            self.tiles_per_frame = 1

    def __len__(self):
        return self.framecount * self.tiles_per_frame

    def __getitem__(self, idx):
        frame_idx = idx // self.tiles_per_frame
        tile_idx = idx % self.tiles_per_frame

        tile_y_idx = tile_idx // self.tiles_x
        tile_x_idx = tile_idx % self.tiles_x

        frame = self.source.get_image(frame_idx)  # numpy array

        if self.tile_size:
            y0 = tile_y_idx * self.tile_size
            x0 = tile_x_idx * self.tile_size
            y1 = min(y0 + self.tile_size, self.img_h)
            x1 = min(x0 + self.tile_size, self.img_w)
            frame = frame[y0:y1, x0:x1, :]  # crop region

        # TODO: Should labels not always be 0? How do I build my test dataset? CVAT Instance?
        out = {"image": frame, "label": 0}  # label=0 for "good"

        if self.transform:
            out = self.transform(**out)

        # Should the image be transposed?
        # frame = frame.transpose(2, 0, 1)

        # Normalize?
        #frame = frame.float() / 255.0

        return out


class VideoDataModule(LightningDataModule):
    def __init__(self, train_dataset, test_dataset=None, batch_size=8):
        # TODO: Support transforms

        super().__init__()
        self.train_dataset = train_dataset
        if test_dataset is None:
            warnings.warn("No test dataset provided, using the train dataset as test...")
            # TODO: Should this behavior even exist, for debugging for now
            self.test_dataset = self.train_dataset
        else:
            self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
