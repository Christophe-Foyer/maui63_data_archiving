# %%

from pycocotools.coco import COCO
import os
import numpy as np
from PIL import Image
import torch
import skimage.draw

import time
from functools import wraps
from pycocotools import mask as coco_mask


def timer(threshold=0.1):
    """Only log if function takes longer than threshold seconds"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            if elapsed > threshold:
                if args and hasattr(args[0].__class__, "__name__"):
                    class_name = args[0].__class__.__name__
                    print(f"SLOW: {class_name}.{func.__name__} took {elapsed:.3f}s")
                else:
                    print(f"SLOW: {func.__name__} took {elapsed:.3f}s")

            return result

        return wrapper

    return decorator


# @timer(threshold=0.5)
# def ann_to_mask(segmentation, h, w):
#     mask = np.zeros((h, w), dtype=np.uint8)

#     # COCO segmentation is a list of polygons [[x1, y1, x2, y2...], ...]
#     # Your data seems to have it wrapped in a numpy array, handling both:
#     if isinstance(segmentation, np.ndarray):
#         segmentation = segmentation.tolist()

#     for poly in segmentation:
#         # Reshape to (N, 2)
#         poly_coords = np.array(poly).reshape(-1, 2)

#         # Get pixel coordinates inside the polygon
#         rr, cc = skimage.draw.polygon(
#             poly_coords[:, 1], poly_coords[:, 0], shape=(h, w)
#         )
#         mask[rr, cc] = 1

#     return mask


@timer(threshold=3)
def ann_to_mask(segmentation, h, w):
    # pycocotools has optimized C code for this
    rle = coco_mask.frPyObjects(segmentation, h, w)
    mask = coco_mask.decode(rle)

    # If multiple polygons, merge them
    if len(mask.shape) == 3:
        mask = mask.max(axis=2)

    return mask.astype(np.uint8)


# %%


import albumentations as A


class CocoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        annotations_file="_annotations.coco.json",
        transforms=None,
        cache_masks=True,
        max_cache_size=0.25,  # So we don't fill up memory too much, store slowest masks (0-1 for percentage)
    ):
        self.dataset_dir = dataset_dir
        self.coco = COCO(os.path.join(dataset_dir, annotations_file))
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

        self.cache_masks = cache_masks

        # Convert float to int based on dataset size
        if max_cache_size <= 1:
            self.max_cache_size = int(max_cache_size * len(self.ids))
        else:
            self.max_cache_size = max_cache_size

        self._mask_cache = {} if cache_masks else None
        self._mask_creation_times = {} if cache_masks else None

    def __len__(self):
        return len(self.ids)

    @timer(threshold=10)
    def __getitem__(self, idx):
        # Image info
        image_id = self.ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.dataset_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Get or create mask
        h, w = image.shape[:2]
        if self.cache_masks:
            if idx in self._mask_cache:
                instance_mask = self._mask_cache[idx]
            else:
                # Time the mask creation
                start = time.perf_counter()
                instance_mask = self._create_mask(image_id, h, w)
                elapsed = time.perf_counter() - start

                # Add to cache
                if len(self._mask_cache) < self.max_cache_size:
                    # Cache not full, just add
                    self._mask_cache[idx] = instance_mask
                    self._mask_creation_times[idx] = elapsed
                else:
                    # Cache full, find fastest item
                    # TODO: Store ordered to reduce compute?
                    min_idx = min(
                        self._mask_creation_times, key=self._mask_creation_times.get
                    )

                    # Only replace if new mask took longer
                    if elapsed > self._mask_creation_times[min_idx]:
                        del self._mask_cache[min_idx]
                        del self._mask_creation_times[min_idx]
                        self._mask_cache[idx] = instance_mask
                        self._mask_creation_times[idx] = elapsed
        else:
            instance_mask = self._create_mask(image_id, h, w)

        result = {"image": image, "id": image_info["id"], "masks": instance_mask}

        if self.transforms is not None:
            result = self._apply_transforms(result, annotations)

        return result

    def _create_mask(self, image_id, h, w):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        instance_mask = np.zeros((h, w), dtype=np.int32)
        for ann in annotations:
            binary_mask = ann_to_mask(ann["segmentation"], h, w)
            instance_mask[binary_mask > 0] = ann["id"]

        return instance_mask

    def _apply_transforms(self, result, annotations):
        image = result["image"]
        instance_mask = result["masks"]

        # Prepare input for albumentations
        transform_inputs = {"image": image, "mask": instance_mask}

        # Introspect transforms to handle bboxes automatically
        bbox_params = None
        if (
            hasattr(self.transforms, "processors")
            and "bboxes" in self.transforms.processors
        ):
            bbox_params = self.transforms.processors["bboxes"].params

            bboxes_list = [ann["bbox"] for ann in annotations]
            category_ids = [ann["category_id"] for ann in annotations]

            if bbox_params.label_fields:
                for field in bbox_params.label_fields:
                    transform_inputs[field] = category_ids
                transform_inputs["bboxes"] = bboxes_list
            else:
                # Append label to bbox if no label_fields
                bboxes_with_labels = []
                for bbox, label in zip(bboxes_list, category_ids):
                    bboxes_with_labels.append(list(bbox) + [label])
                transform_inputs["bboxes"] = bboxes_with_labels

        # Apply transforms
        transformed = self.transforms(**transform_inputs)

        # Update result
        result["image"] = transformed["image"]
        result["masks"] = transformed["mask"]

        if "bboxes" in transformed:
            result["bboxes"] = transformed["bboxes"]

        # Retrieve label fields if present
        if bbox_params and bbox_params.label_fields:
            for field in bbox_params.label_fields:
                if field in transformed:
                    result[field] = transformed[field]

        return result


# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    external_dir = os.path.join(_script_dir, "../../data/external")

    # Define transforms
    transforms = A.Compose(
        [
            A.Resize(800, 800),
            A.RandomRotate90(p=0.5),
            # CenterCrop to test coordinate clipping/transformation
            A.CenterCrop(600, 600),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )

    datasets = []

    # Iterate over directories in data/external
    if os.path.exists(external_dir):
        for dataset_name in os.listdir(external_dir):
            train_dir = os.path.join(external_dir, dataset_name, "train")
            # Check if train dir and annotations exist
            if os.path.isdir(train_dir) and os.path.exists(
                os.path.join(train_dir, "_annotations.coco.json")
            ):
                print(f"Loading dataset: {dataset_name}")
                try:
                    ds = CocoDataset(train_dir, transforms=transforms)
                    if len(ds) > 0:
                        datasets.append(ds)
                except Exception as e:
                    print(f"Failed to load {dataset_name}: {e}")

    if not datasets:
        print("No valid datasets found in data/external")
    else:
        concat_dataset = torch.utils.data.ConcatDataset(datasets)
        print(f"Total samples: {len(concat_dataset)}")

        # Plot a few examples
        num_examples = 5
        indices = np.linspace(0, len(concat_dataset) - 1, num_examples, dtype=int)

        for idx in indices:
            sample = concat_dataset[idx]
            image = sample["image"]
            masks = sample["masks"]

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(image)

            # Plot contours for masks
            if np.max(masks) > 0:
                ax.contour(masks > 0, levels=[0.5], colors="red", linewidths=2)

            # Plot bboxes if present
            if "bboxes" in sample:
                for bbox in sample["bboxes"]:
                    # bbox is [x, y, w, h] for COCO
                    x, y, w, h = bbox
                    # Very thin / dotted / semitransparent bbox
                    rect = patches.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=0.5,
                        edgecolor="white",
                        facecolor="none",
                        linestyle="--",
                        alpha=0.7,
                    )
                    ax.add_patch(rect)

            ax.set_title(f"Sample {idx} (Image ID: {sample['id']})")
            ax.axis("off")
            plt.show()

# %%
