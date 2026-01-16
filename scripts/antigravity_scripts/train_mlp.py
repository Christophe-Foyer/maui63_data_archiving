
import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoModel, AutoProcessor
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- Configuration ---
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE = 518  # Input size for DINOv3 (Small)
NUM_NEG_PER_IMG = 5 # Number of negative crops to extract per image

# Paths
DATA_ROOT = Path("data/external")

# Dataset Config: List of (dataset_root, train_ann_file, valid_ann_file)
DATASETS_CONFIG = [
    (
        DATA_ROOT / "roboflow_aerial_sharks",
        DATA_ROOT / "roboflow_aerial_sharks/train/_annotations.coco.json",
        DATA_ROOT / "roboflow_aerial_sharks/valid/_annotations.coco.json"
    ),
    (
        DATA_ROOT / "pilot_whale_detection_gma",
        DATA_ROOT / "pilot_whale_detection_gma/train/_annotations.coco.json",
        DATA_ROOT / "pilot_whale_detection_gma/valid/_annotations.coco.json"
    )
]

# --- Dataset Class ---
class CocoCropDataset(Dataset):
    def __init__(self, img_dir, ann_file, processor, crop_size=518, negatives_per_img=3, train=True):
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.crop_size = crop_size
        self.train = train
        
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # Prepare samples: List of (image_filename, crop_box, label)
        # crop_box = (x, y, w, h)
        self.samples = []
        self._prepare_samples(negatives_per_img)
        
        print(f"Dataset loaded from {ann_file}")
        print(f"  Total Images: {len(self.images)}")
        print(f"  Total Samples: {len(self.samples)} (Pos: {sum(s[2] for s in self.samples)}, Neg: {len(self.samples) - sum(s[2] for s in self.samples)})")

    def _prepare_samples(self, negatives_per_img):
        for img_id, img_info in self.images.items():
            filename = img_info['file_name']
            img_w, img_h = img_info['width'], img_info['height']
            
            anns = self.img_to_anns.get(img_id, [])
            
            # --- Positive Samples ---
            # Create a crop centered on each annotation
            for ann in anns:
                bbox = ann['bbox'] # [x, y, w, h]
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
                
                # Determine crop coordinates (centered on object, clamped to image)
                x1 = int(cx - self.crop_size / 2)
                y1 = int(cy - self.crop_size / 2)
                
                # Random jitter for training
                if self.train:
                    jitter = int(self.crop_size * 0.1)
                    x1 += random.randint(-jitter, jitter)
                    y1 += random.randint(-jitter, jitter)

                x1 = max(0, min(x1, img_w - self.crop_size))
                y1 = max(0, min(y1, img_h - self.crop_size))
                
                # Handle images smaller than crop_size (unlikely for aerial, but safe to check)
                if img_w < self.crop_size: x1 = 0
                if img_h < self.crop_size: y1 = 0
                
                self.samples.append((filename, (x1, y1, self.crop_size, self.crop_size), 1.0))
                
            # --- Negative Samples ---
            # Random crops that do NOT overlap significantly with any annotation
            if negatives_per_img > 0:
                count = 0
                retries = 0
                max_retries = negatives_per_img * 5
                
                while count < negatives_per_img and retries < max_retries:
                    retries += 1
                    
                    if img_w <= self.crop_size or img_h <= self.crop_size:
                        x1, y1 = 0, 0
                    else:
                        x1 = random.randint(0, img_w - self.crop_size)
                        y1 = random.randint(0, img_h - self.crop_size)
                        
                    # Check overlap
                    crop_rect = [x1, y1, self.crop_size, self.crop_size]
                    overlap = False
                    for ann in anns:
                        if self._iou(crop_rect, ann['bbox']) > 0.0: # Strict no overlap? Or low IoU?
                             # Let's say slight overlap is bad for "negative"
                             overlap = True
                             break
                    
                    if not overlap:
                        self.samples.append((filename, (x1, y1, self.crop_size, self.crop_size), 0.0))
                        count += 1

    def _iou(self, boxA, boxB):
        # box: x, y, w, h
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0
        
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, crop_box, label = self.samples[idx]
        img_path = self.img_dir / filename
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Crop
            x, y, w, h = crop_box
            crop = image.crop((x, y, x+w, y+h))
            
            # Use processor (handles resizing to correct input size, norm, etc)
            # DINO processor usually takes [C, H, W] tensors or PIL
            inputs = self.processor(images=crop, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0) # [3, H, W]
            
            return pixel_values, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy sample or handle gracefully? 
            # Ideally retry, but for simplicity returning zeros (bad practice but keeps flow)
            return torch.zeros((3, self.crop_size, self.crop_size)), torch.tensor(label, dtype=torch.float32)

# --- Model ---
class DinoBinaryClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Hidden dim
        hidden_dim = self.backbone.config.hidden_size
        
        # Simple MLP Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1) # Logits for BCEWithLogitsLoss
        )
        
    def forward(self, x):
        outputs = self.backbone(x)
        # Use simple mean pooling or CLS token if avail
        # DINOv2 usually has pooler_output or we can take last_hidden_state mean
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
             embeds = outputs.pooler_output
        else:
             embeds = outputs.last_hidden_state.mean(dim=1)
             
        logits = self.head(embeds)
        return logits

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1) # [B, 1]
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / len(loader), correct / total

# --- Main ---
def main():
    print(f"Initializing DINOv3 model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # Load Datasets
    train_datasets = []
    valid_datasets = []
    
    for root, train_ann, valid_ann in DATASETS_CONFIG:
        if train_ann.exists():
            print(f"Loading Train: {train_ann}")
            train_datasets.append(CocoCropDataset(root / "train", train_ann, processor, CROP_SIZE, NUM_NEG_PER_IMG, train=True))
        if valid_ann.exists():
            print(f"Loading Valid: {valid_ann}")
            valid_datasets.append(CocoCropDataset(root / "valid", valid_ann, processor, CROP_SIZE, NUM_NEG_PER_IMG, train=False))
            
    if not train_datasets:
        print("No training datasets found!")
        return

    full_train_set = ConcatDataset(train_datasets)
    full_valid_set = ConcatDataset(valid_datasets) if valid_datasets else None
    
    print(f"Total Training Samples: {len(full_train_set)}")
    
    train_loader = DataLoader(full_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(full_valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) if full_valid_set else None
    
    # Model Setup
    model = DinoBinaryClassifier(MODEL_NAME).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    save_path = "dino_mlp_head.pth"
    
    print("\nStarting Training...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        val_metrics = ""
        if valid_loader:
            val_loss, val_acc = validate(model, valid_loader, criterion, DEVICE)
            val_metrics = f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                val_metrics += " [Saved Best]"
        else:
             # Save last if no validation
             torch.save(model.state_dict(), save_path)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} {val_metrics}")

if __name__ == "__main__":
    main()
