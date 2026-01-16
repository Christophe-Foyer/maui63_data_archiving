# %%

from pycocotools.coco import COCO
import os
import numpy as np
from PIL import Image
import torch

import numpy as np
import skimage.draw


def ann_to_mask(segmentation, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # COCO segmentation is a list of polygons [[x1, y1, x2, y2...], ...]
    # Your data seems to have it wrapped in a numpy array, handling both:
    if isinstance(segmentation, np.ndarray):
        segmentation = segmentation.tolist()
        
    for poly in segmentation:
        # Reshape to (N, 2)
        poly_coords = np.array(poly).reshape(-1, 2)
        
        # Get pixel coordinates inside the polygon
        rr, cc = skimage.draw.polygon(poly_coords[:, 1], poly_coords[:, 0], shape=(h, w))
        mask[rr, cc] = 1
        
    return mask

# %%

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, annotations_file="_annotations.coco.json"):
        self.dataset_dir = dataset_dir
        self.coco = COCO(os.path.join(dataset_dir, annotations_file))
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Image info
        image_id = self.ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.dataset_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Get masks
        mask_ids = self.coco.getAnnIds(imgIds=image_id)
        masks = self.coco.loadAnns(mask_ids)

        # Create instance mask
        h, w = image.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.int32)

        for mask in masks:
            binary_mask = ann_to_mask(mask['segmentation'], h, w)
            instance_mask[binary_mask > 0] = mask['id']
        
        return dict(
            image=image,
            id=image_info['id'],
            masks=instance_mask
        )

# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(_script_dir, "../../data/external/pilot_whale_detection_gma/train/")

    dataset = CocoDataset(dataset_dir)
    idx = 12
    print(dataset[idx])

    # Show image
    plt.imshow(dataset[idx]['image'])
    plt.show()

    # Show masks
    plt.imshow(dataset[idx]['masks'])
    plt.show()

# %%