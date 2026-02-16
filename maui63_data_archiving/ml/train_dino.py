import os
import random
import time
from datetime import datetime

import albumentations as A
import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics
import torchvision
from albumentations.core.transforms_interface import DualTransform
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoModel, AutoProcessor

from maui63_data_archiving.dataset import FrameDataset
from maui63_data_archiving.ml.dataset import CocoDataset


class AddSyntheticBlob(DualTransform):
    def __init__(self, p=0.5, always_apply=False):
        super(AddSyntheticBlob, self).__init__(always_apply, p)

    def get_params(self):
        return {
            "center_x": np.random.uniform(0.1, 0.9),
            "center_y": np.random.uniform(0.1, 0.9),
            "radius": np.random.uniform(0.05, 0.15),
            "color": np.random.randint(0, 255, 3).tolist(),
        }

    def apply(self, img, center_x=0, center_y=0, radius=0.1, color=(0, 0, 0), **params):
        h, w = img.shape[:2]
        center = (int(center_x * w), int(center_y * h))
        r = int(radius * min(h, w))
        # Draw blob
        return cv2.circle(img.copy(), center, r, color, -1)

    def apply_to_mask(self, mask, center_x=0, center_y=0, radius=0.1, **params):
        h, w = mask.shape[:2]
        center = (int(center_x * w), int(center_y * h))
        r = int(radius * min(h, w))
        # Add to mask (using 1 to indicate object presence)
        # Using 1 assuming simple binary/indexed mask
        return cv2.circle(mask.copy(), center, r, 1, -1)


class LogFlaggedImagesCallback(pl.Callback):
    def __init__(self, data_path, num_images=16, tile_size=224, threshold=0.5):
        super().__init__()
        self.data_path = data_path
        self.num_images = num_images
        self.tile_size = tile_size
        self.threshold = threshold
        self.dataset = None

    def on_validation_epoch_end(self, trainer, pl_module):
        # Don't run this during the initial validation sanity check
        # if trainer.sanity_checking:
        #     return
        if not os.path.exists(self.data_path):
            return

        if self.dataset is None:
            self.dataset = FrameDataset(self.data_path, tile_size=self.tile_size)

        # Shuffle indices to grab random tiles/frames
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        # We only search through a limited number of images to find flagged ones to avoid long stalls
        search_limit = 200
        flagged_images = []

        # Use tqdm to show progress during the search
        from tqdm import tqdm

        pbar = tqdm(
            total=min(len(indices), search_limit),
            desc="Searching flagged images",
            leave=False,
        )

        # Use a DataLoader with multiple workers to speed up image fetching
        from torch.utils.data import DataLoader, Subset

        search_subset = Subset(self.dataset, indices[:search_limit])
        # Using batch_size for faster inference and num_workers for faster loading
        loader = DataLoader(search_subset, batch_size=8, num_workers=8, pin_memory=True)

        pl_module.eval()
        with torch.no_grad():
            for batch in loader:
                images_batch = batch["image"]  # (B, H, W, C)
                # Convert to list of numpy for processor
                imgs_list = [img.numpy() for img in images_batch]

                # Batch transform for model
                inputs = pl_module.processor(images=imgs_list, return_tensors="pt")
                inputs = {k: v.to(pl_module.device) for k, v in inputs.items()}

                logits = pl_module(inputs)
                probs = torch.sigmoid(logits).squeeze(-1)  # (B,)

                for j in range(len(probs)):
                    if probs[j] > self.threshold:
                        # Convert to tensor (C, H, W) and normalize to [0, 1]
                        img_tensor = images_batch[j].permute(2, 0, 1).float() / 255.0
                        flagged_images.append(img_tensor)
                        pbar.set_postfix(flagged=len(flagged_images))

                    if len(flagged_images) >= self.num_images:
                        break

                pbar.update(len(images_batch))
                if len(flagged_images) >= self.num_images:
                    break
        pbar.close()

        if flagged_images:
            grid = torchvision.utils.make_grid(
                flagged_images, nrow=int(np.sqrt(self.num_images))
            )
            grid_np = grid.permute(1, 2, 0).cpu().numpy()

            if trainer.logger:
                try:
                    # Try the standard Lightning logger interface first
                    if hasattr(trainer.logger, "log_image"):
                        trainer.logger.log_image(
                            key="flagged_detections",
                            image=grid_np,
                            step=trainer.global_step,
                        )
                    elif hasattr(trainer.logger, "experiment") and hasattr(
                        trainer.logger.experiment, "log_image"
                    ):
                        # Fallback for MLFlowLogger's underlying client
                        trainer.logger.experiment.log_image(
                            run_id=trainer.logger.run_id,
                            image=grid_np,
                            artifact_file=f"flagged_detections/step_{trainer.global_step:010}.png",
                        )
                    else:
                        print("Warning: Logger does not support log_image.")

                    print(
                        f"Logged {len(flagged_images)} flagged images to MLflow at step {trainer.global_step}."
                    )
                except Exception as e:
                    print(f"Failed to log image to MLflow: {e}")


class DinoV3LightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        lr=1e-4,
        pos_weight=1.0,
        unfreeze_last_layer=False,
        verbose=True,
    ):
        super().__init__()

        start_init = time.time()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.unfreeze_last_layer = unfreeze_last_layer
        self.verbose = verbose

        if self.verbose:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Loading backbone: {model_name}..."
            )

        # Load Model
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        if self.verbose:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Backbone loaded in {time.time() - start_init:.2f}s"
            )

        # Handle freezing/unfreezing
        if unfreeze_last_layer:
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze last layer blocks if possible
            # Trying standard ViT / DINOv2 structure
            try:
                # model.encoder.layer is the ModuleList
                # We unfreeze the last block
                if hasattr(self.model, "encoder") and hasattr(
                    self.model.encoder, "layer"
                ):
                    for param in self.model.encoder.layer[-1].parameters():
                        param.requires_grad = True
                    # Also norm
                    if hasattr(self.model.encoder, "norm"):
                        for param in self.model.encoder.norm.parameters():
                            param.requires_grad = True
                elif hasattr(self.model, "layernorm"):
                    for param in self.model.layernorm.parameters():
                        param.requires_grad = True
                print("Unfroze last layer of backbone.")
            except Exception as e:
                print(f"Warning: Could not unfreeze last layer automatically: {e}")
        else:
            # Freeze everything
            for param in self.model.parameters():
                param.requires_grad = False

        # MLP Head for classification
        hidden_dim = self.model.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 1),
        )

        # Use pos_weight to trade off precision for recall
        # Higher positive weight = higher recall, lower precision
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")

    def forward(self, inputs):
        # inputs is the output of processor (dict with pixel_values etc)
        # inputs should already be on device

        # Backbone execution
        if self.unfreeze_last_layer:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        # Use global pooled representation or mean of patches
        features = outputs.last_hidden_state[:, 1:, :].mean(dim=1)

        # TODO: Apply sigmoid (only if not training since BCELoss includes it)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)

        # Calculate metrics
        preds = torch.sigmoid(logits)

        # Update metrics state but don't log per step
        self.accuracy.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.f1.update(preds, labels)

        # Log loss per step is okay, cheap
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute and log metrics once per epoch
        self.log("val_acc", self.accuracy.compute(), prog_bar=True)
        self.log("val_precision", self.precision.compute(), prog_bar=True)
        self.log("val_recall", self.recall.compute(), prog_bar=True)
        self.log("val_f1", self.f1.compute(), prog_bar=True)

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def test_step(self, batch, batch_idx):
        # TODO: Make this also compute only for epochs and only update on step?
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)

        preds = torch.sigmoid(logits)
        self.log("test_loss", loss)
        self.log("test_acc", self.accuracy(preds, labels))
        self.log("test_precision", self.precision(preds, labels))
        self.log("test_recall", self.recall(preds, labels))
        self.log("test_f1", self.f1(preds, labels))
        return loss

    def configure_optimizers(self):
        # Only pass trainable parameters
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.05)

        # Use Cosine Annealing to smoothly decay the learning rate
        # We use the max_epochs from the trainer if available
        max_epochs = self.trainer.max_epochs if self.trainer is not None else 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class CocoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=4,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        verbose=True,
        tile_size=224,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing CocoDataModule (processor: {model_name})...")

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.train_transforms = A.Compose(
            [
                # Crop/Pad FIRST to minimize work for all subsequent steps
                A.PadIfNeeded(min_height=tile_size, min_width=tile_size),
                A.RandomCrop(width=tile_size, height=tile_size),
                AddSyntheticBlob(p=0.2),  # Now only draws on the small patch
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                ),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.GaussianBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.2),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.3,
                ),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
            ]
        )
        self.val_transforms = A.Compose(
            [
                A.PadIfNeeded(min_height=tile_size, min_width=tile_size),
                A.RandomCrop(width=tile_size, height=tile_size),
            ]
        )

    def setup(self, stage=None):
        import time

        start_setup = time.time()
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Setting up datasets...")

        self.train_dataset = self._load_dataset_split("train")
        self.val_dataset = self._load_dataset_split("valid")
        self.test_dataset = self._load_dataset_split("test")

        if self.verbose:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Setup complete in {time.time() - start_setup:.2f}s"
            )
            print(
                f"Train samples: {len(self.train_dataset) if self.train_dataset else 0}"
            )
            print(f"Valid samples: {len(self.val_dataset) if self.val_dataset else 0}")
            print(f"Test samples: {len(self.test_dataset) if self.test_dataset else 0}")

    def _load_dataset_split(self, split_name):
        datasets = []
        if os.path.exists(self.data_dir):
            subdirs = os.listdir(self.data_dir)
            for i, dataset_name in enumerate(subdirs):
                split_dir = os.path.join(self.data_dir, dataset_name, split_name)
                # Check for annotations file
                if os.path.isdir(split_dir) and os.path.exists(
                    os.path.join(split_dir, "_annotations.coco.json")
                ):
                    if self.verbose:
                        print(
                            f"  [{split_name}] Loading {dataset_name} ({i + 1}/{len(subdirs)})...",
                            end="\r",
                        )
                    try:
                        # Select transforms based on split
                        transforms = (
                            self.train_transforms
                            if split_name == "train"
                            else self.val_transforms
                        )

                        ds = CocoDataset(split_dir, transforms=transforms)
                        if len(ds) > 0:
                            datasets.append(ds)
                    except Exception as e:
                        print(f"Skipping {dataset_name}/{split_name}: {e}")

        if datasets:
            return ConcatDataset(datasets)
        return []

    def collate_fn(self, batch):
        images = [item["image"] for item in batch]
        # Label is 1 if any object mask is present, 0 otherwise
        labels = [1.0 if np.max(item["masks"]) > 0 else 0.0 for item in batch]

        inputs = self.processor(images=images, return_tensors="pt")

        return inputs, torch.tensor(labels).unsqueeze(1)

    def train_dataloader(self):
        if not self.train_dataset:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=16,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=16,
        )

    def test_dataloader(self):
        if not self.test_dataset:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=16,
        )


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming data is at ../../data/external relative to this script
    external_dir = os.path.join(_script_dir, "../../data/external")

    pl.seed_everything(42)
    # model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
    tile_size = 512

    # 1. Init DataModule
    dm = CocoDataModule(data_dir=external_dir, batch_size=4, model_name=model_name, tile_size=tile_size)

    # 2. Init Model
    # pos_weight=5.0 to penalize false negatives 5x more than false positives
    # This increases recall (catch more positives) at the cost of precision (more false alarms)
    model = DinoV3LightningModule(
        model_name=model_name,
        pos_weight=5.0,
        unfreeze_last_layer=True,
    )

    # 3. Loggers
    # By default, Lightning logs to TensorBoard (save_dir="lightning_logs")
    # We make it explicit here so you can easily change it or swap to MLFlow
    # from lightning.pytorch.loggers import TensorBoardLogger
    # logger = TensorBoardLogger("tb_logs", name="dinov3_classifier")

    # If you want to use MLFlow instead:
    from lightning.pytorch.loggers import MLFlowLogger

    logger = MLFlowLogger(
        experiment_name="dinov3_classifier",
        # tracking_uri="file:./mlruns",
        tracking_uri="http://localhost:5000",
    )

    # 4. Callbacks
    from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # TQDMProgressBar(refresh_rate=1),
    ]
    maui_data_path = "data/maui63/maui63_images"
    if os.path.exists(maui_data_path):
        callbacks.append(
            LogFlaggedImagesCallback(
                data_path=maui_data_path, num_images=16, threshold=0.5, tile_size=tile_size
            )
        )
        print(f"Added LogFlaggedImagesCallback using {maui_data_path}")

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        log_every_n_steps=5,
        val_check_interval=1 / 2,  # 2 times per epoch
        logger=logger,
        callbacks=callbacks,
    )

    # 6. Fit
    print("Starting training...")
    trainer.fit(model, datamodule=dm)

    # 7. Save Model State (Optional, for compatibility with inference scripts)
    save_path = os.path.join(
        _script_dir,
        f"dinov3_classifier_pl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
    )
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
