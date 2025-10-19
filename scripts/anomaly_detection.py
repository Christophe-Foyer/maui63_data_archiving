# %%

import os
from torch.utils.data.dataset import ConcatDataset
from maui63_data_archiving.dataset import VideoFrameDataset, VideoDataModule
import albumentations as A
import cv2

# %%

def get_concat_dataset(folder):
    files = [os.path.join(folder, file) for file in os.listdir(folder)]

    return ConcatDataset([
        VideoFrameDataset(
            file,

            # TODO: Each video has a different scale / zoom so we probably want to adjust this so the tiles are about the size we want
            # Less of an issue once we start using maui data maybe? Though different altitude flights might cause scale issues, maybe it shouldn't be sensitive to scale
            tile_size=1024,

            # Resize the tiles to a specific shape
            transform=A.Compose(
                [
                    A.Resize(
                        height=256,
                        width=256,
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        p=1.0
                    )
                ],
            )
        ) for file in files
    ])

data = VideoDataModule(
    train_dataset=get_concat_dataset("../test_data/train_videos"),
    test_dataset=get_concat_dataset("../test_data/test_videos"),
)

# %%

import matplotlib.pyplot as plt

plt.imshow(data.test_dataset.datasets[1][0]["image"])
plt.show()

plt.imshow(data.test_dataset.datasets[1][1]["image"])
plt.show()

# %%

# WIP

# %%
