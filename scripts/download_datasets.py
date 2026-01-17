
import os
import argparse
import requests
import json
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/external")

def apply_autodistill_segmentation(dataset_path):
    """
    Applies SAM (Segment Anything Model) to generate segmentation masks 
    for bounding boxes in the dataset using HuggingFace Transformers.
    """
    print(f"\n[Auto-Distill] Starting SAM-based segmentation for {dataset_path}...")
    try:
        import torch
        import numpy as np
        from transformers import SamModel, SamProcessor
        from pycocotools.coco import COCO
        from skimage import measure
        from PIL import Image
    except ImportError as e:
        print(f"[!] Auto-Distill skipped: {e}")
        print("    Please ensure 'transformers', 'torch', 'scikit-image', 'pycocotools' are installed.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Auto-Distill] Loading facebook/sam-vit-base on {device}...")
    
    try:
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    except Exception as e:
        print(f"[!] Failed to load SAM model: {e}")
        return

    # Find JSONs
    annotation_files = list(Path(dataset_path).glob("**/*.json"))
    # Filter for likely COCO files
    annotation_files = [f for f in annotation_files if "coco" in f.name.lower() or "annotations" in f.name.lower()]
    
    if not annotation_files:
        print(f"[!] No annotation files found in {dataset_path}.")
        return

    for ann_file in annotation_files:
        print(f"[Auto-Distill] Processing {ann_file.name}...")
        try:
            # Load JSON content
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            # Map images
            img_dict = {img['id']: img for img in coco_data.get('images', [])}
            
            # Group annotations by image
            from collections import defaultdict
            img_to_anns = defaultdict(list)
            for ann in coco_data.get('annotations', []):
                img_to_anns[ann['image_id']].append(ann)
            
            # Parent directory for images
            img_dir = ann_file.parent
            
            modified_count = 0
            
            for img_id, anns in tqdm(img_to_anns.items(), desc=f"Segmenting {ann_file.name}"):
                # Filter annotations that need segmentation
                # We check if 'segmentation' is missing or has only 0 points or matches bbox exactly (rare for COCO)
                anns_to_process = []
                for ann in anns:
                    seg = ann.get('segmentation', [])
                    # In COCO RLE, counts is a list. In Polygon, it's list of lists.
                    # If empty or checks fail:
                    if not seg: 
                         anns_to_process.append(ann)
                    elif isinstance(seg, list) and len(seg) == 0:
                         anns_to_process.append(ann)
                    # If we wanted to check if segmentation is "bad" (e.g. 4 points box), we could.
                    # But user asked "if it doesn't already have masks".
                
                if not anns_to_process:
                    continue

                img_info = img_dict.get(img_id)
                if not img_info: continue
                
                # Try finding the image
                img_fname = img_info.get('file_name')
                img_path = img_dir / img_fname
                if not img_path.exists():
                     # Fallback to simple name
                     img_path = img_dir / Path(img_fname).name
                
                if not img_path.exists():
                    continue

                try:
                    raw_image = Image.open(img_path).convert("RGB")
                    
                    # Prepare boxes for this image
                    # SAM expects [[x1, y1, x2, y2]] structure
                    # COCO bbox: [x, y, w, h] => [x, y, x+w, y+h]
                    input_boxes = []
                    for ann in anns_to_process:
                        bbox = ann.get('bbox', [])
                        if len(bbox) == 4:
                            input_boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                        else:
                            input_boxes.append([0, 0, 1, 1]) # dummy

                    if not input_boxes: continue

                    inputs = processor(
                        raw_image, 
                        input_boxes=[input_boxes], 
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Post process
                    masks = processor.image_processor.post_process_masks(
                        outputs.pred_masks.cpu(), 
                        inputs["original_sizes"].cpu(), 
                        inputs["reshaped_input_sizes"].cpu()
                    )
                    # masks is list of shape (batch, num_boxes, num_masks, height, width)
                    # num_masks usually 3 (scores). We take best score idx? 
                    # By default Transformers implementation usually returns 3 masks. 
                    # model output pred_iou_scores can help, but typically index 0 is best or index 0 is valid?
                    # Actually standard SAM returns 3 masks, index 0 is typically the one to use or we check scores.
                    # outputs.iou_scores is (batch, num_boxes, num_masks)
                    
                    scores = outputs.iou_scores.cpu().numpy()[0] # (num_boxes, 3)
                    batch_masks = masks[0] # (num_boxes, 3, H, W)

                    for i, ann in enumerate(anns_to_process):
                        # Pick best mask based on IoU score
                        best_mask_idx = np.argmax(scores[i])
                        best_mask = batch_masks[i, best_mask_idx].numpy() # boolean mask

                        # Find contours (skimage)
                        contours = measure.find_contours(best_mask, 0.5)
                        
                        segmentation = []
                        for contour in contours:
                            # (row, col) -> (y, x) -> want (x, y)
                            contour = np.flip(contour, axis=1)
                            # flatten
                            segmentation.append(contour.flatten().tolist())
                        
                        ann['segmentation'] = segmentation
                        modified_count += 1
                        
                except Exception as e:
                    print(f"Error processing image {img_fname}: {e}")

            if modified_count > 0:
                print(f"Updated {modified_count} annotations in {ann_file.name}. Saving...")
                # Backup
                backup_file = ann_file.with_suffix('.bak.json')
                if not backup_file.exists():
                     with open(backup_file, 'w') as f:
                        pass # relying on atomic write or just overwrite safety isn't huge here.
                             # Actually we should write the original content to backup if it's new.
                             # But we've parsed it.
                
                with open(ann_file, 'w') as f:
                    json.dump(coco_data, f)
            else:
                 print(f"No annotations needed update in {ann_file.name}.")

        except Exception as e:
            print(f"[!] Error processing {ann_file}: {e}")

    print(f"[Auto-Distill] Completed {dataset_path}.\n")


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    with open(dest_path, "wb") as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_zenodo_record(record_id, output_dir):
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching metadata from {api_url}...")
    
    try:
        r = requests.get(api_url)
        if r.status_code != 200:
            print(f"Failed to get Zenodo record {record_id}. Status: {r.status_code}")
            return
        
        data = r.json()
        record_dir = output_dir / f"zenodo_{record_id}"
        record_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Found dataset: {data.get('metadata', {}).get('title', 'Unknown')}")
        
        for file_info in data.get('files', []):
            file_url = file_info['links']['self']
            filename = file_info['key']
            dest = record_dir / filename
            
            if dest.exists():
                print(f"File {filename} already exists. Skipping.")
            else:
                print(f"Downloading {filename}...")
                download_file(file_url, dest)
                
            # Extract if archive
            if filename.endswith(".zip"):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(dest, 'r') as zip_ref:
                    zip_ref.extractall(record_dir)
            elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
                print(f"Extracting {filename}...")
                with tarfile.open(dest, "r:gz") as tar:
                    tar.extractall(record_dir)
                    
        print(f"Successfully downloaded Zenodo record {record_id}")
        return record_dir
    except Exception as e:
        print(f"Error downloading Zenodo record: {e}")

def download_roboflow_dataset(api_key, projects, format="yolov8", apply_autodistill=False):
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)

        for proj_config in projects:
            workspace = proj_config["workspace"]
            project_name = proj_config["project"]
            version = proj_config["version"]
            output_name = proj_config.get("output_dir_name", f"roboflow_{project_name}")
            
            print(f"\nAttempting to download Roboflow dataset '{project_name}' (Version {version})...")
            try:
                project = rf.workspace(workspace).project(project_name)
                dataset = project.version(version).download(format, location=str(DATA_DIR / output_name))
                print(f"Successfully downloaded {project_name}.")
                
                if apply_autodistill:
                    apply_autodistill_segmentation(DATA_DIR / output_name)
            except Exception as e_proj:
                print(f"[!] Failed to download project '{project_name}': {e_proj}")

    except ImportError:
        print("\n[!] 'roboflow' library not installed. Install with `pip install roboflow` to use this feature.")
    except Exception as e:
        print(f"\n[!] Failed to initialize Roboflow: {e}")
        print("Make sure your API key is correct.")

def main():
    parser = argparse.ArgumentParser(description="Download and optionally process ML datasets.")
    parser.add_argument("--autodistill", action="store_true", help="Apply SAM-based auto-distillation (segmentation) to Roboflow datasets.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Zenodo: SharkTrack Training Dataset - Record ID 15625845
    # print("--- 1. Downloading Zenodo Datasets ---")
    # download_zenodo_record("15625845", DATA_DIR)
    
    # 2. Roboflow
    print("\n--- 2. Downloading Roboflow Datasets ---")
    
    # Configuration for Roboflow projects
    roboflow_projects = [{
            "workspace": path.split("/")[0],
            "project": path.split("/")[1],
            "version": 1, # TODO: Any datasets have more than 1 version?
            "output_dir_name": path.split("/")[1]
        } for path in [
            "elec440/aerial-shark-images",
            "computer-vision-xnbbv/pilot_whale_detection_gma",
            "thesis-urxei/marlin",
            "xupeng/ocean_val",
            "wavec-offshore-renewables/dolphin-detection-qn12h",
            "cetaceans/project1-w5dg4",
            "labgym-2f8fe/white-beaked-dolphins",
        ]
    ]

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key:
        download_roboflow_dataset(api_key, roboflow_projects, format="coco", apply_autodistill=args.autodistill)
    else:
        print("Skipping Roboflow download. Set 'ROBOFLOW_API_KEY' environment variable to download.")
        print("You can get a key from https://app.roboflow.com/")
        for p in roboflow_projects:
             print(f"Target Dataset: https://universe.roboflow.com/{p['workspace']}/{p['project']}")

    # # 3. Manual Instructions
    # print("\n--- 3. Other Datasets (Manual Download Required) ---")
    # print("NOAA ASAMM (Arctic Marine Mammals):")
    # print("  - Requires contacting NOAA/Dr. Megan Ferguson.")
    # print("  - URL: https://www.fisheries.noaa.gov/alaska/marine-mammal-protection/aerial-surveys-arctic-marine-mammals")
    
    # print("\nWhales from Space:")
    # print("  - Requires account/email.")
    # print("  - URL: https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=90fab89e-5d07-4d5c-b619-60799a4d09f8")

if __name__ == "__main__":
    main()
