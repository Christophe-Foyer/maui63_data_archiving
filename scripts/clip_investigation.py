# %%

# import albumentations as A
# import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from maui63_data_archiving.dataset import FrameDataset

# %%

def find_matching_indices(
    dataset,
    text_prompts=("a dolphin", "a seagull", "an animal in the ocean"),
    model_name="openai/clip-vit-large-patch14",
    batch_size=16,
    top_k=200,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeds = []

    # Encode images
    print("Processing images...")
    for batch in tqdm(dataloader, desc="Extracting CLIP embeddings"):
        imgs = [Image.fromarray(im_batch.numpy()) for im_batch in batch["image"]]
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embeds = model.get_image_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu())

    image_embeds = torch.cat(all_embeds, dim=0)

    # Encode text prompts
    print("Encoding prompts...")
    text_inputs = processor(text=list(text_prompts), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Compute similarities
    print("Computing similarities...")
    sims = image_embeds.to(device).matmul(text_embeds.T.to(device))  # (N_images, N_prompts)
    avg_scores = sims.mean(dim=1).cpu().numpy()  # Average similarity across prompts

    # Select top-k
    top_indices = np.argsort(avg_scores)[::-1]
    if top_k:
        top_indices = top_indices[:top_k]

    return top_indices, avg_scores

# %%

if __name__ == "__main__":
    print("Loading dataset...")

    # dataset = FrameDataset(
    #     "../test_data/maui63_images",
    
    #     # TODO: Each video has a different scale / zoom so we probably want to adjust this so the tiles are about the size we want
    #     # Less of an issue once we start using maui data maybe? Though different altitude flights might cause scale issues, maybe it shouldn't be sensitive to scale
    #     tile_size=256,

    #     # Resize the tiles to a specific shape
    #     # transform=A.Compose(
    #     #     [
    #     #         A.Resize(
    #     #             height=256,
    #     #             width=256,
    #     #             interpolation=cv2.INTER_LINEAR,
    #     #             mask_interpolation=cv2.INTER_NEAREST,
    #     #             p=1.0
    #     #         )
    #     #     ],
    #     # )
    # )
    
    # Dolphin video
    dataset = FrameDataset("../test_data/test_videos/277315.mp4", tile_size=1024)
    
    # To use a subset
    dataset = torch.utils.data.Subset(dataset=dataset, indices=range(64))

    print("Finding tiles matching prompts...")
    top_indices, avg_scores = find_matching_indices(
        dataset, top_k=10,
        text_prompts = [
            "an animal"
            "a dolphin",
        ]
    )

# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # just show first 10
    for idx in top_indices[:10]:
        img = dataset[idx]["image"]
        plt.imshow(img)
        plt.title(f"Index: {idx}")
        plt.show()

# %%
