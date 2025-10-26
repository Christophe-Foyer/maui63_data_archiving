# %%

import torch
from transformers import AutoModel, AutoProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from maui63_data_archiving.dataset import FrameDataset


# %%

def compute_embeddings(dataset, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    model.eval()
    all_embeds = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(loader, desc="Extracting DINOv3 embeddings"):
        images = [Image.fromarray(im_batch.numpy()) for im_batch in batch["image"]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

            # use global pooled representation
            embeds = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(1)

        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)


def find_similar_images(dataset_embeds, reference_embeds, top_k=100):
    # compute cosine similarity between dataset and reference embeddings
    sims = dataset_embeds.matmul(reference_embeds.T)

    # best match per dataset image
    max_sim, _ = sims.max(dim=1)

    top_indices = torch.topk(max_sim, top_k).indices.numpy()
    return top_indices, max_sim.numpy()

# %%

# Note: This is my dataset video: 
# https://pixabay.com/videos/drone-flying-camera-4k-drone-277315/

if __name__ == '__main__':
    full_dataset = FrameDataset(
        # "../test_data/test_videos/277315.mp4",
        "../test_data/maui63_images",
        tile_size=1024,
    )
    full_dataset = torch.utils.data.Subset(
        full_dataset,
        # indices=range(128),

        # Lazy approximation to fetch only images with dolphins in the real dataset
        indices=range(int(len(full_dataset)*0.34), int(len(full_dataset)*0.44)),
    )

    reference_dataset = torch.utils.data.Subset(
        dataset=FrameDataset(
            "../test_data/test_videos/277315.mp4",
            tile_size=1024,
        ),
        indices=[149, 150, 151, 170, 135]
    )

    # reference_dataset and full_dataset both return {"image": np.array(...)}
    ref_embeds = compute_embeddings(reference_dataset)
    data_embeds = compute_embeddings(full_dataset)

# %%

if __name__ == "__main__":
    top_indices, scores = find_similar_images(data_embeds, ref_embeds, top_k=50)

# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # just show first 20
    for idx in top_indices[:20]:
        img = full_dataset[idx]["image"]
        plt.imshow(img)
        plt.title(f"Index: {idx}")
        plt.show()

# %% Tile plots

if __name__ == "__main__":
    import math

    n_images = min(20, len(top_indices))
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axes = axes.flatten()

    # plot images
    for i, idx in enumerate(top_indices[:n_images]):
        img = full_dataset[idx]["image"]
        axes[i].imshow(img)
        axes[i].set_title(f"Index: {idx}")
        axes[i].axis('off')  # Hide axes for cleaner look

    # hide any unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# %%

if __name__ == "__main__":

    n_images = len(reference_dataset)
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axes = axes.flatten()

    # plot images
    for i, idx in enumerate(range(n_images)):
        img = reference_dataset[idx]["image"]
        axes[i].imshow(img)
        axes[i].set_title(f"Index: {idx}")
        axes[i].axis('off')  # Hide axes for cleaner look

    # hide any unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# %%
