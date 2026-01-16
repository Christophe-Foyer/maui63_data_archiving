
# %% Imports

import sys
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

from maui63_data_archiving.dataset import FrameDataset
from maui63_data_archiving.detection import FewShotDetector
from transformers import AutoProcessor

# Import the model class from train_mlp (or duplicate it if preferred to stay self-contained)
# For simplicity, I'll redefine the minimal class here or try to import if in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from train_mlp import DinoBinaryClassifier, MODEL_NAME, CROP_SIZE
except ImportError:
    # Fallback if cannot import
    MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
    CROP_SIZE = 518
    import torch.nn as nn
    from transformers import AutoModel
    class DinoBinaryClassifier(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden_dim = self.backbone.config.hidden_size
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 1)
            )
        def forward(self, x):
            outputs = self.backbone(x)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                 embeds = outputs.pooler_output
            else:
                 embeds = outputs.last_hidden_state.mean(dim=1)
            logits = self.head(embeds)
            return logits

# %% Setup Datasets

# Training Data (Video)
# We train on the video where we have specific example indices
# Training Data (Video) - OPTIONAL for MLP, Required for Few-Shot
train_video_path = os.path.join(Path(__file__).parent, "../test_data/test_videos/277315.mp4")
dataset_train_full = None
if os.path.exists(train_video_path):
    print(f"Loading training data from {train_video_path}")
    dataset_train_full = FrameDataset(train_video_path, tile_size=1024)
else:
    print(f"Warning: Training video not found at {train_video_path}. Few-shot similarity will be skipped.")

# Inference Data (Images)
# We want to find tiles in the image folder
inference_image_path = os.path.join(Path(__file__).parent, "../test_data/maui63_images")
if not os.path.exists(inference_image_path):
    raise FileNotFoundError(f"Inference image path not found at {inference_image_path}")
    
print(f"Loading inference data from {inference_image_path}")
dataset_inference = FrameDataset(inference_image_path, tile_size=1024)


# %% Prepare Training Data

# Define Indices ( Global indices relative to dataset_train_full )
pos_indices = [149, 150, 151, 170, 135]

# Specific negative indices (User provided)
# Add any specific tile indices you want to treat as negatives here.
neg_indices_fixed = [] 

# Ensure indices are valid
max_idx = len(dataset_train_full)
pos_indices = [i for i in pos_indices if i < max_idx]
neg_indices_fixed = [i for i in neg_indices_fixed if i < max_idx]

# Random negatives to fill the pool
num_random_negatives = 10
candidates = set(range(min(1000, max_idx))) - set(pos_indices) - set(neg_indices_fixed)

if len(candidates) < num_random_negatives:
    neg_indices_random = list(candidates)
else:
    neg_indices_random = random.sample(list(candidates), num_random_negatives)

all_neg_indices = neg_indices_fixed + neg_indices_random

# Create a Subset for Training
# We only want to compute embeddings for the few-shot examples, not the whole video
train_subset_indices = pos_indices + all_neg_indices
dataset_train_subset = torch.utils.data.Subset(dataset_train_full, train_subset_indices)

# Map original global indices to local subset indices for the trainer
# The subset maps 0..N to the indices in train_subset_indices.
# So first items are positives, next are negatives.
local_pos_indices = list(range(len(pos_indices)))
local_neg_indices = list(range(len(pos_indices), len(pos_indices) + len(all_neg_indices)))

print(f"Training with {len(local_pos_indices)} positives and {len(local_neg_indices)} negatives.")


# %% Initialize Detector
detector = FewShotDetector()

query_embedding = None
top_indices = []
top_scores = []

if dataset_train_full:
    # Compute embeddings for positive examples only
    # We need these to form the query
    print("Computing embeddings for positive examples...")
    # Create a subset for just the positive indices
    dataset_pos = torch.utils.data.Subset(dataset_train_full, pos_indices)
    pos_embeddings = detector.compute_embeddings(dataset_pos)

    # Compute mean embedding for query
    query_embedding = pos_embeddings.mean(dim=0)


# %% Inference

# Compute embeddings for the target inference dataset
# For speed in testing, we can limit this if the folder is huge
MAX_INFERENCE_FRAMES = 500
if len(dataset_inference) > MAX_INFERENCE_FRAMES:
    print(f"Subsetting inference to first {MAX_INFERENCE_FRAMES} tiles for speed...")
    inference_indices = range(MAX_INFERENCE_FRAMES)
    dataset_inference_run = torch.utils.data.Subset(dataset_inference, inference_indices)
else:
    dataset_inference_run = dataset_inference

print("Computing embeddings for inference dataset...")
inference_embeddings = detector.compute_embeddings(dataset_inference_run)

# %% Predict / Get Top K
# Predict / Get Top K
if query_embedding is not None:
    top_indices, top_scores = detector.search_by_similarity(inference_embeddings, query_embedding, k=5)

print("\nTop 5 Detections:")
for idx, score in zip(top_indices, top_scores):
    print(f"Index: {idx}, Score: {score:.4f}")
    
# %% Use MLP Classifier (Optional)
from tqdm.auto import tqdm

USE_MLP = True
CHECKPOINT_PATH = os.path.join(Path(__file__).parent, "../dino_mlp_head.pth")

if USE_MLP and os.path.exists(CHECKPOINT_PATH):
    print(f"\nLoading trained MLP classifier from {CHECKPOINT_PATH}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Init Model
    model = DinoBinaryClassifier(MODEL_NAME).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # Run Inference on same subset
    print("Running MLP inference...")
    mlp_scores = []
    mlp_indices = []
    
    # Prepare batch inference for speed? Or simple loop
    # Simple loop for demonstration
    with torch.no_grad():
        for i in tqdm(range(len(dataset_inference_run))):
            # Warning: dataset_inference_run returns 'image' as numpy array
            # We need to process it for the model
            img_np = dataset_inference_run[i]["image"]
            inputs = processor(images=img_np, return_tensors="pt").to(device)
            
            logits = model(inputs.pixel_values)
            prob = torch.sigmoid(logits).item()
            
            mlp_scores.append(prob)
            mlp_indices.append(inference_indices[i] if 'inference_indices' in locals() else i)

    # Top K from MLP
    mlp_scores = np.array(mlp_scores)
    mlp_indices = np.array(mlp_indices)
    
    top_k_idx = mlp_scores.argsort()[-5:][::-1]
    top_indices_mlp = mlp_indices[top_k_idx]
    top_scores_mlp = mlp_scores[top_k_idx]
    
    print("\nTop 5 MLP Detections:")
    for idx, score in zip(top_indices_mlp, top_scores_mlp):
        print(f"Index: {idx}, Score: {score:.4f}")
        
    # Visualize MLP Results
    fig, axes = plt.subplots(1, 5, figsize=(50, 10))
    fig.suptitle("MLP Classifier Results", fontsize=24)
    for i, idx in enumerate(top_indices_mlp):
        img = dataset_inference_run[idx if 'inference_indices' not in locals() else list(inference_indices).index(idx)]["image"]
        axes[i].imshow(img)
        axes[i].set_title(f"Idx: {idx}\nProb: {top_scores_mlp[i]:.4f}")
        axes[i].axis('off')
    plt.show()

else:
    print("\nSkipping MLP inference (Use 'train_mlp.py' to create checkpoint first)")

# %% Visualize Similarity Results (Existing)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

print("\nVisualizing Similarity Search Results...")
fig, axes = plt.subplots(1, 5, figsize=(50, 10))
fig.suptitle("Similarity Search Results", fontsize=24)
for i, idx in enumerate(top_indices):
    img = dataset_inference_run[idx if 'inference_indices' not in locals() else list(inference_indices).index(idx)]["image"]
    axes[i].imshow(img)
    axes[i].set_title(f"Idx: {idx}\nSim: {top_scores[i]:.2f}")
    axes[i].axis('off')
plt.show()

# %%