# %%

from pycocotools.coco import COCO
import os
import numpy as np
import torch
import time
from functools import wraps
from pycocotools import mask as coco_mask
import json
from pathlib import Path
import cv2
import heapq


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
        # Min-heap to track the fastest items currently in our "Top Slowest" cache
        # Stores (creation_time, idx)
        self._mask_priority_heap = [] if cache_masks else None

    def __len__(self):
        return len(self.ids)

    @timer(threshold=10)
    def __getitem__(self, idx):
        # Image info
        image_id = self.ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.dataset_dir, image_info["file_name"])

        # OpenCV is typically faster than PIL for decoding and NumPy conversion
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations once
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
                instance_mask = self._create_mask_optimized(annotations, h, w)
                elapsed = time.perf_counter() - start

                # Add to cache using a Min-Heap to maintain the "Slowest K" items
                # Overhead: O(log N) to insert/evict instead of O(N)
                if len(self._mask_cache) < self.max_cache_size:
                    self._mask_cache[idx] = instance_mask
                    heapq.heappush(self._mask_priority_heap, (elapsed, idx))
                else:
                    # Peek at the fastest mask currently in our cache
                    fastest_in_cache_time, fastest_idx = self._mask_priority_heap[0]

                    if elapsed > fastest_in_cache_time:
                        # The new mask is slower than our fastest cached mask, swap them
                        heapq.heapreplace(self._mask_priority_heap, (elapsed, idx))
                        del self._mask_cache[fastest_idx]
                        self._mask_cache[idx] = instance_mask
                    # else: new mask is faster than everything in cache, don't store it
        else:
            instance_mask = self._create_mask_optimized(annotations, h, w)

        result = {"image": image, "id": image_info["id"]}
        if self.cache_masks:
            result["masks"] = instance_mask

        # Add basic detection info (will be overwritten by transforms if they exist)
        result["bboxes"] = [ann["bbox"] for ann in annotations]
        result["category_ids"] = [ann["category_id"] for ann in annotations]

        if self.transforms is not None:
            result = self._apply_transforms(result, annotations)

        return result

    def _create_mask_optimized(self, annotations, h, w):
        """Batch processed mask creation using pycocotools C-optimizations."""
        instance_mask = np.zeros((h, w), dtype=np.int32)
        if not annotations:
            return instance_mask

        # Filter out annotations without segmentation
        segmentations = [
            ann["segmentation"] for ann in annotations if "segmentation" in ann
        ]
        valid_anns = [ann for ann in annotations if "segmentation" in ann]

        if not segmentations:
            return instance_mask

        # pycocotools can batch process segmentations
        rles = coco_mask.frPyObjects(segmentations, h, w)
        masks = coco_mask.decode(rles)  # Returns (h, w, n)

        if len(masks.shape) == 3:
            for i, ann in enumerate(valid_anns):
                m = masks[:, :, i]
                instance_mask[m > 0] = ann["id"]
        else:
            # Single annotation case
            instance_mask[masks > 0] = valid_anns[0]["id"]

        return instance_mask

    def _create_mask(self, image_id, h, w):
        # Kept for backward compatibility but deprecated
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        return self._create_mask_optimized(annotations, h, w)

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


def merge_coco_datasets(dataset_dirs, output_dir):
    """
    Merges multiple COCO datasets into one.
    Assumes each directory has a 'train', 'valid', and 'test' subdirectory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_out_dir = output_dir / split
        split_out_dir.mkdir(exist_ok=True)
        images_out_dir = split_out_dir

        merged_images = []
        merged_annotations = []
        merged_categories = []
        cat_name_to_new_id = {}

        image_id_offset = 0
        ann_id_offset = 0

        for ds_dir in dataset_dirs:
            ds_dir = Path(ds_dir)
            split_dir = ds_dir / split
            ann_file = split_dir / "_annotations.coco.json"

            if not ann_file.exists():
                continue

            with open(ann_file, "r") as f:
                coco_data = json.load(f)

            # 1. Normalize Categories
            current_ds_cat_id_to_new_id = {}
            for cat in coco_data.get("categories", []):
                name = cat["name"]
                if name not in cat_name_to_new_id:
                    # Create new category in merged list
                    new_id = len(merged_categories) + 1  # COCO usually uses 1-based IDs
                    cat_name_to_new_id[name] = new_id
                    new_cat = cat.copy()
                    new_cat["id"] = new_id
                    merged_categories.append(new_cat)

                current_ds_cat_id_to_new_id[cat["id"]] = cat_name_to_new_id[name]

            # 2. Images
            img_id_map = {}
            for img in coco_data.get("images", []):
                old_id = img["id"]
                new_id = old_id + image_id_offset
                img_id_map[old_id] = new_id

                # Copy or symlink image
                src_path = split_dir / img["file_name"]
                dst_name = f"{ds_dir.name}_{img['file_name']}"
                dst_path = images_out_dir / dst_name

                if src_path.exists():
                    # Use unique name to avoid collisions
                    img["file_name"] = dst_name
                    img["id"] = new_id
                    if not dst_path.exists():
                        os.symlink(os.path.abspath(src_path), dst_path)
                    merged_images.append(img)

            # 3. Annotations
            for ann in coco_data.get("annotations", []):
                ann["id"] = ann["id"] + ann_id_offset
                ann["image_id"] = img_id_map.get(ann["image_id"])
                # Update category_id using the map
                ann["category_id"] = current_ds_cat_id_to_new_id.get(
                    ann["category_id"], ann["category_id"]
                )
                if ann["image_id"] is not None:
                    merged_annotations.append(ann)

            # Increment offsets (using max ids to be safe)
            if coco_data.get("images"):
                image_id_offset = max([img["id"] for img in merged_images]) + 1
            if coco_data.get("annotations"):
                ann_id_offset = max([ann["id"] for ann in merged_annotations]) + 1

        if merged_images:
            merged_coco = {
                "images": merged_images,
                "annotations": merged_annotations,
                "categories": merged_categories,
            }
            with open(split_out_dir / "_annotations.coco.json", "w") as f:
                json.dump(merged_coco, f)
            print(
                f"Split '{split}': Merged {len(merged_images)} images and {len(merged_annotations)} annotations."
            )


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
