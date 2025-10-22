# %%

import os
from torch.utils.data.dataset import ConcatDataset, Subset
from maui63_data_archiving.dataset import FrameDataset
import albumentations as A
import cv2

# %% Create dataset

# TODO: Should I provide a script to download the data automatically?
# Maybe it's easier to just use maui63 data.
# One challenge with maui63 data is making sure we don't feed any data with objects

def get_concat_dataset(folder):
    files = [os.path.join(folder, file) for file in os.listdir(folder)]

    return ConcatDataset([
        FrameDataset(
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

# TODO: Some of these videos do have people/seagulls in them, those should probably be flagged
train_dataset = get_concat_dataset("../test_data/train_videos")
test_dataset = get_concat_dataset("../test_data/test_videos")

# %% Plot samples

import matplotlib.pyplot as plt

# plt.imshow(train_dataset.datasets[1][0]["image"])
# plt.show()

# plt.imshow(train_dataset.datasets[1][1]["image"])
# plt.show()

# %% Output dataset to folders

from anomalib.data import Folder
from tqdm.auto import tqdm
import numpy as np

# Set up a subset
max_samples = 150
train_subset = Subset(
    dataset=train_dataset,
    indices=np.random.choice(
        len(train_dataset),
        size=min(max_samples, len(train_dataset)),
        replace=False
    )
)

# Output the dataset to a folder
# TODO: Nuke folders ahead of time
# TODO: Get actual labels in the train dataset
# TODO: Mask_dir?
anomalib_root = "../test_data/anomalib/train"
no_obj_folder = os.path.join(anomalib_root, "no_object")
obj_folder = os.path.join(anomalib_root, "object")
os.makedirs(anomalib_root, exist_ok=True)
os.makedirs(no_obj_folder, exist_ok=True)
os.makedirs(obj_folder, exist_ok=True)

# %%

# for idx, row in tqdm(enumerate(train_subset), total=len(train_subset)):
#     if row["label"] == 0:
#         cv2.imwrite(os.path.join(no_obj_folder, f"{idx}.png"), row["image"])
#     else:
#         cv2.imwrite(os.path.join(obj_folder, f"{idx}.png"), row["image"])

# %%

for idx in [2105, 2106, 2107]:
    row = test_dataset[idx]
    # plt.imshow(row["image"])
    cv2.imwrite(os.path.join(obj_folder, f"{idx}.png"), row["image"])

# %%

# Create the datamodule
datamodule = Folder(
    name="animal_detector",
    root=anomalib_root,
    normal_dir="no_object",
    abnormal_dir="object",
    # task="classification",
    train_batch_size=32,
    eval_batch_size=1,
)

# Setup the datamodule
datamodule.setup()

# %%

# from anomalib.visualization import ImageVisualizer
from anomalib.models import Patchcore
from anomalib.engine import Engine

# visualizer = ImageVisualizer()
model = Patchcore(
    # visualizer=visualizer
)
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)

# %%

# Create the datamodule
test_datamodule = Folder(
    name="animal_detector",
    root=anomalib_root,
    normal_dir="object",
    abnormal_dir="object",
    # task="classification",
    train_batch_size=1,
    eval_batch_size=1,
)

# Setup the datamodule
test_datamodule.setup()

from anomalib.visualization.image.item_visualizer import visualize_image_item
from anomalib.data import ImageItem

predictions = engine.predict(
    datamodule=test_datamodule,
    model=model,
)

# Generate visualization
for i in range(len(predictions)):
    pred = predictions[i]
    item = ImageItem(
        image=pred["image"][0],
        image_path=pred["image_path"][0],
        pred_mask=pred["pred_mask"][0],
        pred_label=pred["pred_label"][0],
    )
    vis_result = visualize_image_item(item, fields=["image", "pred_mask"])
    print("Is_Anomaly:", pred["pred_label"][0])
    plt.imshow(vis_result)
    plt.show()

# %%
