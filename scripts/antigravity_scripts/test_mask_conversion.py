
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import skimage.draw

def polygons_to_mask_array(masks_list, height, width):
    """
    Converts a list of COCO-style mask annotations to an N-d array.
    
    Args:
        masks_list: List of annotation dicts.
        height: Image height.
        width: Image width.
        
    Returns:
        A dictionary containing:
        - 'instance_mask': (H, W) array where pixels are object IDs.
        - 'binary_stack': (N, H, W) binary array, one layer per object.
    """
    
    # Initialize containers
    instance_mask = np.zeros((height, width), dtype=np.int32)
    binary_stack = []
    
    for ann in masks_list:
        obj_id = ann['id']
        seg = ann['segmentation']
        
        # Handle the user's specific format where segmentation might be a numpy array
        if isinstance(seg, np.ndarray):
            # If it's (1, N) or similar, flatten it to list
            seg = seg.tolist()
        
        # COCO segmentation allows list of lists (multiple polygons for one object)
        # or RLE. The user example implies polygons.
        
        # Check if polygon or RLE
        if isinstance(seg, list):
            # Polygon
            # We can use pycocotools if available, but let's show an skimage version 
            # so it works without the COCO object if needed, though COCO.annToMask is simpler.
            
            # Using skimage.draw.polygon for each polygon in the list
            # COCO polygons are [x1, y1, x2, y2, ...]
            # rle = maskUtils.frPyObjects(seg, height, width)
            # m = maskUtils.decode(rle)
            # m = np.sum(m, axis=2) > 0 # Combine multiple polygons
            
            # Alternative: direct rasterization (simpler/std python)
            curr_mask = np.zeros((height, width), dtype=np.uint8)
            
            for poly in seg:
                poly = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(poly[:, 1], poly[:, 0], shape=(height, width))
                curr_mask[rr, cc] = 1
                
        elif isinstance(seg, dict) and 'counts' in seg:
            # RLE
            curr_mask = maskUtils.decode(seg)
        else:
             # Fallback if unsure
             continue
             
        # Add to instance mask (pixels = ID)
        # Note: overlapping objects will overwrite each other in the instance mask.
        instance_mask[curr_mask > 0] = obj_id
        
        binary_stack.append(curr_mask)
        
    binary_stack = np.array(binary_stack)
    
    return instance_mask, binary_stack

# Test with the user's data
if __name__ == "__main__":
    # Mock data based on user request
    # NOTE: The coordinates in the user example are around 200-400. 
    # Assumed image size must cleanly contain them.
    H, W = 500, 500
    
    sample_masks = [{'id': 1,
      'image_id': 0,
      'category_id': 1,
      'bbox': [210, 371, 11.75, 72.296],
      'area': 849.481,
      'segmentation': np.array([[210.25 , 386.844, 211.75 , 388.741, 212.125, 395.615, 213.   ,
              399.17 , 213.25 , 406.519, 215.625, 427.852, 215.875, 436.859,
              215.125, 439.467, 212.25 , 443.496, 221.375, 443.496, 221.25 ,
              441.126, 217.875, 437.333, 217.75 , 421.926, 219.125, 407.23 ,
              219.125, 404.148, 218.5  , 402.015, 219.75 , 399.17 , 219.625,
              395.378, 218.125, 392.533, 218.25 , 387.556, 219.25 , 383.526,
              222.   , 383.052, 221.25 , 381.156, 219.   , 379.733, 217.875,
              377.837, 217.25 , 373.807, 215.875, 371.2  , 214.125, 371.2  ,
              212.375, 374.044, 212.5  , 379.733, 212.   , 382.104, 210.5  ,
              384.711, 210.25 , 386.844]]),
      'iscrowd': 0}]
      
    instance_array, binary_array = polygons_to_mask_array(sample_masks, H, W)
    
    print(f"Created instance array of shape: {instance_array.shape}")
    print(f"Unique values in instance array: {np.unique(instance_array)}")
    print(f"Created binary stack of shape: {binary_array.shape}")
