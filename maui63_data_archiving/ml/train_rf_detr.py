import os
import json
import random
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import torch
import torchvision
import mlflow
from tqdm import tqdm
from PIL import Image
import supervision as sv

from rfdetr import RFDETRBase
from maui63_data_archiving.dataset import FrameDataset

import warnings

# Filter annoying Torch/Library warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*non-tuple sequence for multidimensional indexing.*",
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_coco_dataset(
    input_json: str, images_dir: str, output_dir: str, train_ratio=0.8, val_ratio=0.1
):
    """
    Splits a single COCO dataset into train, valid, and test sets.
    Ensures categories have 'supercategory' for RF-DETR compatibility.
    """
    output_dir = Path(output_dir)
    if (output_dir / "train" / "_annotations.coco.json").exists():
        logger.info(f"Dataset already split in {output_dir}. Skipping split.")
        return

    logger.info(f"Splitting dataset {input_json}...")
    with open(input_json, "r") as f:
        data = json.load(f)

    images = data["images"]
    img_id_to_anns = {img["id"]: [] for img in images}
    for ann in data["annotations"]:
        # Ensure COCO compliance
        if "area" not in ann:
            # bbox is [x, y, w, h]
            ann["area"] = ann["bbox"][2] * ann["bbox"][3]
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
        if "segmentation" not in ann:
            ann["segmentation"] = []

        img_id_to_anns[ann["image_id"]].append(ann)

    random.seed(42)
    random.shuffle(images)

    n = len(images)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)

    splits = {
        "train": images[:train_n],
        "valid": images[train_n : train_n + val_n],
        "test": images[train_n + val_n :],
    }

    # RF-DETR requires supercategory != "none" to load classes
    categories = data["categories"]
    for cat in categories:
        if "supercategory" not in cat or cat["supercategory"] == "none":
            cat["supercategory"] = "animal"

    for split_name, split_images in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_anns = []
        for img in tqdm(split_images, desc=f"Processing {split_name}"):
            split_anns.extend(img_id_to_anns[img["id"]])

            # Symlink image to the split directory
            src = Path(images_dir) / img["file_name"]
            dst = split_dir / img["file_name"]

            if src.exists():
                if not dst.exists():
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copy2(src, dst)
            else:
                logger.warning(f"Image not found: {src}")

        split_data = {
            "images": split_images,
            "annotations": split_anns,
            "categories": categories,
        }

        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(split_data, f)

    logger.info(f"Dataset split completed. Output in {output_dir}")


class RFDETRMLFlowCallback:
    """
    Custom callback for RF-DETR to log metrics and visual detections to MLflow.
    """

    def __init__(
        self,
        model_wrapper,
        experiment_name: str,
        tracking_uri: str,
        data_path: str,
        tile_size=640,
        threshold=0.3,
        search_limit=200,
        num_flagged_to_log=16,
    ):
        self.model_wrapper = model_wrapper
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.data_path = data_path
        self.tile_size = tile_size
        self.threshold = threshold
        self.search_limit = search_limit
        self.num_flagged_to_log = num_flagged_to_log
        self.dataset = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def on_fit_epoch_end(self, log_stats: Dict[str, Any]):
        epoch = log_stats.get("epoch", 0)

        # 1. Log metrics
        metrics = {}
        for k, v in log_stats.items():
            if isinstance(v, (int, float, np.number)):
                metrics[k.replace("test_coco_eval_bbox", "metrics/mAP")] = float(v)

        # Standardize some keys for easier plotting
        if "test_coco_eval_bbox" in log_stats:
            ce = log_stats["test_coco_eval_bbox"]
            metrics["val/mAP"] = float(ce[0])
            metrics["val/mAP_50"] = float(ce[1])

        mlflow.log_metrics(metrics, step=epoch)

        # 2. visual evaluation on tiled dataset
        if not os.path.exists(self.data_path):
            return

        if self.dataset is None:
            try:
                self.dataset = FrameDataset(self.data_path, tile_size=self.tile_size)
            except Exception as e:
                logger.warning(f"Could not load evaluation dataset: {e}")
                return

        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        flagged_tiles = []

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        logger.info(
            f"Epoch {epoch}: Searching for detections in tiled dataset (limit: {self.search_limit}, threshold: {self.threshold})..."
        )

        # Ensure model is in eval mode for visual assessment
        self.model_wrapper.model.model.eval()

        pbar = tqdm(total=self.search_limit, desc=f"Epoch {epoch} search", leave=False)
        for idx in indices[: self.search_limit]:
            try:
                sample = self.dataset[idx]
                image_np = sample["image"]  # H, W, C (RGB)

                # RFDETR.predict returns sv.Detections
                detections = self.model_wrapper.predict(
                    image_np, threshold=self.threshold
                )

                if isinstance(detections, list):
                    detections = detections[0]

                if len(detections.xyxy) > 0:
                    # Annotate image
                    annotated_img = box_annotator.annotate(
                        scene=image_np.copy(), detections=detections
                    )
                    labels = [f"score: {conf:.2f}" for conf in detections.confidence]
                    annotated_img = label_annotator.annotate(
                        scene=annotated_img, detections=detections, labels=labels
                    )

                    img_tensor = (
                        torch.from_numpy(annotated_img).permute(2, 0, 1).float() / 255.0
                    )
                    flagged_tiles.append(img_tensor)
                    pbar.set_postfix(flagged=len(flagged_tiles))

                if len(flagged_tiles) >= self.num_flagged_to_log:
                    break
            except Exception as e:
                logger.warning(f"Error processing visual tile {idx}: {e}")
                continue
            finally:
                pbar.update(1)
        pbar.close()

        if flagged_tiles:
            logger.info(
                f"Epoch {epoch}: Flagged {len(flagged_tiles)} tiles with detections. Logging to MLflow."
            )
            grid = torchvision.utils.make_grid(
                flagged_tiles, nrow=4, padding=4, pad_value=1.0
            )
            grid_np = grid.permute(1, 2, 0).cpu().numpy()

            # log_image expects HWC
            mlflow.log_image(grid_np, artifact_file=f"detections/epoch_{epoch:03}.png")
        else:
            logger.info(
                f"Epoch {epoch}: No detections found in {self.search_limit} tiles."
            )

    def on_train_batch_start(self, batch_stats):
        pass

    def on_train_end(self):
        logger.info("Training ended.")


def main():
    """
    Trains an RF-DETR model using Maui63 dolphin dataset.
    """
    # 1. Paths setup
    script_dir = Path(__file__).parent.absolute()
    base_dir = script_dir.parent.parent

    # Input dataset (Maui63 provided dataset)
    maui_dolphins_dir = base_dir / "data" / "maui63" / "202508-dolphins"
    input_json = maui_dolphins_dir / "all.json"
    images_source = maui_dolphins_dir / "all"

    # Output split dataset
    processed_dir = base_dir / "data" / "processed"
    split_dir = processed_dir / "maui63_dolphins_split"

    # Tiled evaluation data (as used in train_dino.py)
    maui_eval_path = base_dir / "data" / "maui63" / "maui63_images"

    if not input_json.exists():
        logger.error(f"Input annotations not found at {input_json}")
        return

    # 2. Split dataset for training
    split_coco_dataset(str(input_json), str(images_source), str(split_dir))

    # 3. Initialize RF-DETR-Base model
    # Switching to RFDETRBase as it uses patch_size=14 which more closely matches DINOv2 defaults.
    # Note: Pretrained weights are loaded from rf-detr-base.pth, which includes
    # the DINOv2 backbone fine-tuned for detection.
    logger.info("Initializing RF-DETR-Base model...")
    model = RFDETRBase()

    # 4. Setup MLflow and Callbacks
    experiment_name = "maui63_rfdetr_dolphins"
    tracking_uri = "http://localhost:5000"

    mlflow_callback = RFDETRMLFlowCallback(
        model_wrapper=model,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        data_path=str(maui_eval_path)
        if maui_eval_path.exists()
        else str(images_source),
        tile_size=640,
        threshold=0.3,
        search_limit=200,
        num_flagged_to_log=16,
    )

    # Add callbacks directly to the model instance
    model.callbacks["on_fit_epoch_end"].append(mlflow_callback.on_fit_epoch_end)
    model.callbacks["on_train_batch_start"].append(mlflow_callback.on_train_batch_start)
    model.callbacks["on_train_end"].append(mlflow_callback.on_train_end)

    # 5. Start training
    logger.info("Starting training with MLflow logging...")

    with mlflow.start_run():
        mlflow.log_param("model_type", "rf-detr-base")
        mlflow.log_param("dataset", "maui63_202508-dolphins")

        try:
            model.train(
                dataset_dir=str(split_dir),
                dataset_file="roboflow",
                epochs=100,
                imgsz=560,  # RF-DETR-Base default resolution
                batch=4,  # Base is larger than Small, reducing batch size to avoid OOM
                grad_accum_steps=4,
                project="maui63_det",
                name="rf-detr-base_dolphins",
                exist_ok=True,
            )
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


if __name__ == "__main__":
    main()
