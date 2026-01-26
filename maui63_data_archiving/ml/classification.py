# %%

from transformers import AutoModel, AutoProcessor
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import torch

# %%


class FewShotClassifier:
    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        batch_size=32,
        device=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute_embeddings(self, dataset):
        model = AutoModel.from_pretrained(self.model_name).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_name)

        model.eval()
        all_embeds = []
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for batch in tqdm(loader, desc="Extracting DINOv3 embeddings"):
            images = [Image.fromarray(im_batch.numpy()) for im_batch in batch["image"]]
            inputs = processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)

                # use global pooled representation
                embeds = (
                    outputs.pooler_output
                    if hasattr(outputs, "pooler_output")
                    else outputs.last_hidden_state.mean(1)
                )

            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0)

    def find_similar_images(
        dataset_embeds, reference_embeds, top_k=-1, use_reference_mean=True
    ):
        if use_reference_mean:
            # compute cosine similarity between dataset and mean of reference embeddings
            mean_ref_embeds = reference_embeds.mean(dim=0)
            mean_ref_embeds = mean_ref_embeds / mean_ref_embeds.norm(
                dim=-1
            )  # Re-normalize mean vector
            sims = dataset_embeds.matmul(mean_ref_embeds)
            max_sim = sims
        else:
            # compute cosine similarity between dataset and reference embeddings
            sims = dataset_embeds.matmul(reference_embeds.T)

            # best match per dataset image
            max_sim, _ = sims.max(dim=1)

        if top_k > 0:
            top_indices = torch.topk(max_sim, top_k).indices.numpy()
        else:
            top_indices = torch.topk(max_sim, len(max_sim)).indices.numpy()
        return top_indices, max_sim.numpy()

    def train_classifier(self, dataset):
        # Assumes the dataset label key contains an actual class value
        pass


class DinoV3Classifier(torch.nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        batch_size=32,
        device=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Freeze backbone to allow gradient flow for GradCAM (unlike torch.no_grad())
        self.model.requires_grad_(False)

        # MLP Head for classification
        hidden_dim = self.model.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 1),
        )

    def forward(self, x):
        # Handle list of images (inference/training with raw images)
        if isinstance(x, list):
            x = self.processor(images=x, return_tensors="pt").to(self.device)
        # Handle Tensor (pixel_values directly, e.g. for GradCAM)
        elif isinstance(x, torch.Tensor):
            x = {"pixel_values": x}

        # x is now a dict/BatchEncoding compatible with **x
        # We removed torch.no_grad() to support GradCAM, but model is frozen in __init__
        x = self.model(**x)
        x = x.last_hidden_state[:, 1:, :].mean(
            dim=1
        )  # Should this be the mean of [1:]? I think this is based on patch embeddings
        return self.head(x)


# %% Simple Train Loop

if __name__ == "__main__":
    import os
    import numpy as np
    import albumentations as A
    from maui63_data_archiving.ml.dataset import CocoDataset
    import torch.optim as optim

    # Setup paths
    # Assuming running from the repo root
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    external_dir = os.path.join(_script_dir, "../../data/external")

    # 1. Define transforms for training
    transforms = A.Compose(
        [
            A.PadIfNeeded(min_height=512, min_width=512),
            A.RandomCrop(width=512, height=512),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    )

    # 2. Setup Dataset
    datasets = []
    if os.path.exists(external_dir):
        for dataset_name in os.listdir(external_dir):
            train_dir = os.path.join(external_dir, dataset_name, "train")
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
        print("No datasets found. Exiting.")
    else:
        full_dataset = torch.utils.data.ConcatDataset(datasets)
        print(f"Total training samples: {len(full_dataset)}")

        # 3. DataLoader & Collate Function
        def collate_fn(batch):
            images = [item["image"] for item in batch]
            # Label is 1 if any object mask is present, 0 otherwise
            labels = [1.0 if np.max(item["masks"]) > 0 else 0.0 for item in batch]
            return images, torch.tensor(labels).unsqueeze(1)  # (B, 1)

        dataloader = DataLoader(
            full_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=12,
        )

        # 4. Model, Optimizer, Loss
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        classifier = DinoV3Classifier(device=device).to(device)
        optimizer = optim.AdamW(classifier.parameters(), lr=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        # 5. Training Loop
        model_save_path = os.path.join(_script_dir, "dinov3_classifier.pth")

        if os.path.exists(model_save_path):
            print(f"Loading existing model from {model_save_path}")
            classifier.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            num_epochs = 10
            classifier.train()

            print("Starting training...")
            for epoch in range(num_epochs):
                total_loss = 0
                num_batches = 0

                pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
                for images, labels in pbar:
                    labels = labels.to(device)

                    # Forward
                    outputs = classifier(images)
                    loss = criterion(outputs, labels)

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

            print("Training complete.")
            torch.save(classifier.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

# %%

if __name__ == "__main__":
    from maui63_data_archiving.dataset import FrameDataset
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    num_sample = 500
    threshold = 0.3

    # 6. Evaluation on Tiled Dataset
    tiled_dir = os.path.join(_script_dir, "../../data/maui63/maui63_images")
    if os.path.exists(tiled_dir):
        print(f"Running evaluation on tiled dataset: {tiled_dir}")
        try:
            # Tile size 512 matches training crop.
            # Assuming labels are not useful for inference/flagging.
            dataset_tiled = FrameDataset(tiled_dir, tile_size=512)

            # Use a random subset for evaluation
            if len(dataset_tiled) > 0:
                subset_size = min(len(dataset_tiled), num_sample)
                indices = torch.randperm(len(dataset_tiled)).tolist()[:subset_size]
                dataset_tiled = torch.utils.data.Subset(dataset_tiled, indices)
                print(f"Using subset of {subset_size} tiles for evaluation.")

            def eval_collate(batch):
                return [item["image"] for item in batch]

            eval_loader = DataLoader(
                dataset_tiled,
                batch_size=8,
                shuffle=False,
                collate_fn=eval_collate,
                num_workers=12,
            )

            classifier.eval()
            print("\n--- Evaluation Results ---")

            # Setup PDF and GradCAM
            pdf_path = os.path.join(_script_dir, "evaluation_report.pdf")
            print(f"Saving evaluation report to {pdf_path}")

            # Define reshape transform for DINOv2/ViT (B, N, D) -> (B, D, H, W)
            # Assuming 512x512 with patch 16 -> 32x32 grid
            def reshape_transform(tensor):
                # Exclude CLS token if present (usually index 0) and reshape
                # tensor: B, 1025, D -> Spatial 32x32
                result = tensor[:, 1:, :].reshape(
                    tensor.size(0), 32, 32, tensor.size(2)
                )
                # (B, H, W, D) -> (B, D, H, W)
                result = result.transpose(2, 3).transpose(1, 2)
                return result

            # Target layer: Last block norm of the encoder
            # DINOv2 structure usually: model.encoder.layer[-1].norm1
            target_layers = [classifier.model.encoder.layer[-1].norm1]
            cam = GradCAM(
                model=classifier,
                target_layers=target_layers,
                reshape_transform=reshape_transform,
            )

            global_idx = 0

            with PdfPages(pdf_path) as pdf:
                # Iterate batches
                for batch_images in tqdm(eval_loader, desc="Evaluating Tiled Data"):
                    # Pre-process to get tensors for GradCAM
                    # classifier.forward(list) handles this, but we need the tensor for cam.
                    # We manually process here to pass tensor to both.
                    inputs = classifier.processor(
                        images=batch_images, return_tensors="pt"
                    ).to(device)
                    pixel_values = inputs["pixel_values"]

                    # Compute GradCAM
                    # Target class 0 (since single output, index 0)
                    targets = [
                        ClassifierOutputTarget(0) for _ in range(pixel_values.shape[0])
                    ]
                    grayscale_cams = cam(input_tensor=pixel_values, targets=targets)

                    # Get Model Outputs (scores)
                    # We reuse pixel_values so we don't re-process
                    outputs = classifier(pixel_values)
                    scores = torch.sigmoid(outputs)

                    if scores.ndim == 0:
                        scores = scores.unsqueeze(0)

                    for i, score in enumerate(scores):
                        val = score.item()
                        is_flagged = val > threshold
                        img_np = batch_images[i]

                        # Visualization
                        visualization = show_cam_on_image(
                            img_np / 255.0, grayscale_cams[i, :], use_rgb=True
                        )

                        # Create Figure
                        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                        ax.imshow(visualization)
                        title = f"Idx: {global_idx + i} | Flagged: {is_flagged} | Score: {val:.4f}"
                        if is_flagged:
                            title = ">>> " + title
                        ax.set_title(title, color="red" if is_flagged else "black")
                        ax.axis("off")

                        pdf.savefig(fig)
                        plt.close(fig)

                        if is_flagged:
                            print(f"Flagged Img {global_idx + i}: {val:.4f}")

                    global_idx += len(batch_images)

        except Exception as e:
            print(f"Failed to evaluate on tiled dataset: {e}")
            import traceback

            traceback.print_exc()

# %%
