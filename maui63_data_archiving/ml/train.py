import os
import torch
import numpy as np
import cv2
import random
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import torchmetrics
import lightning.pytorch as pl
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoModel, AutoProcessor
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


class DinoV3LightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        lr=1e-4,
        pos_weight=1.0,
        unfreeze_last_layer=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.unfreeze_last_layer = unfreeze_last_layer

        # Load Model
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

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
        return torch.optim.AdamW(params, lr=self.lr)


class CocoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=4,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.train_transforms = A.Compose(
            [
                AddSyntheticBlob(p=0.2),  # Fake "positive" blobs
                A.PadIfNeeded(min_height=224, min_width=224),
                A.RandomCrop(width=224, height=224),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.GaussianBlur(p=0.3),
            ]
        )
        self.val_transforms = A.Compose(
            [
                A.PadIfNeeded(min_height=224, min_width=224),
                A.RandomCrop(width=224, height=224),
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = self._load_dataset_split("train")
        self.val_dataset = self._load_dataset_split("valid")
        self.test_dataset = self._load_dataset_split("test")

        print(f"Train samples: {len(self.train_dataset) if self.train_dataset else 0}")
        print(f"Valid samples: {len(self.val_dataset) if self.val_dataset else 0}")
        print(f"Test samples: {len(self.test_dataset) if self.test_dataset else 0}")

    def _load_dataset_split(self, split_name):
        datasets = []
        if os.path.exists(self.data_dir):
            for dataset_name in os.listdir(self.data_dir):
                split_dir = os.path.join(self.data_dir, dataset_name, split_name)
                # Check for annotations file
                if os.path.isdir(split_dir) and os.path.exists(
                    os.path.join(split_dir, "_annotations.coco.json")
                ):
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
            num_workers=12,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=12,
        )

    def test_dataloader(self):
        if not self.test_dataset:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=12,
        )


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming data is at ../../data/external relative to this script
    external_dir = os.path.join(_script_dir, "../../data/external")

    pl.seed_everything(42)

    # 1. Init DataModule
    dm = CocoDataModule(data_dir=external_dir, batch_size=4)

    # 2. Init Model
    # pos_weight=5.0 to penalize false negatives 5x more than false positives
    # This increases recall (catch more positives) at the cost of precision (more false alarms)
    model = DinoV3LightningModule(
        pos_weight=5.0,
        unfreeze_last_layer=True,
    )

    # 3. Trainer
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

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        # log_every_n_steps=5,
        val_check_interval=0.25,  # Run validation 4 times per epoch
        logger=logger,
    )

    # 4. Fit
    print("Running initial validation check...")
    trainer.validate(model, datamodule=dm)

    print("Starting training...")
    trainer.fit(model, datamodule=dm)

    # 5. Save Model State (Optional, for compatibility with inference scripts)
    save_path = os.path.join(_script_dir, "dinov3_classifier_pl.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
