import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    """Helper function to visualize the mask."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Helper function to visualize the input points."""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def main():
    # --- Configuration ---
    # The checkpoint path has been updated to the absolute path requested.
    checkpoint_path = "/home/ekrem/Dev/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    # Input and Output Directories
    input_folder = "GeomDir"
    output_folder = "GeomDir/segmented"

    # --- Check for Checkpoint ---
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        print(f"Please ensure the path is correct and the file exists.")
        sys.exit(1)

    # --- Check Input Folder ---
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        print(f"Please create a folder named '{input_folder}' and put your images inside.")
        sys.exit(1)

    # --- Create Output Folder ---
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder '{output_folder}' is ready.")

    # --- Device Selection ---
    # Uses CUDA if available (which you installed), otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    # We load the model once, outside the loop
    print("Loading SAM model... this might take a moment.")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # --- Process Images ---
    # List all image files in the input folder
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"No images found in '{input_folder}'.")
        return

    print(f"Found {len(image_files)} images to process.")

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        print(f"\nProcessing '{image_file}'...")

        # --- Load Image ---
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Warning: Could not load image {image_file}. Skipping.")
            continue
            
        # SAM requires the image to be in RGB format
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions to set a safe center point prompt
        h, w = image.shape[:2]
        
        predictor.set_image(image)

        # --- Prompting ---
        # We provide a single point prompt at the center of the loaded image.
        input_point = np.array([[w // 2, h // 2]]) 
        input_label = np.array([1]) # 1 indicates a foreground point

        # predict returns masks, quality scores, and low resolution logits
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # --- Visualization ---
        # We usually get 3 masks (different ambiguity levels). We'll save the best one.
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]

        print(f"  Best mask score: {best_score:.3f}")

        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_mask(best_mask, plt.gca())
        # # show_points(input_point, input_label, plt.gca())
        # # plt.title(f"Result for {image_file}\nScore: {best_score:.3f}", fontsize=18)
        # plt.axis('off')
        
        # # Save to output folder
        # output_filename = os.path.join(output_folder, f"segmented_{image_file}")
        # plt.savefig(output_filename)
        # print(f"  Saved to '{output_filename}'")
        # plt.close()

        # 1. Convert mask to uint8 (0 or 255) for OpenCV
        binary_mask = (best_mask * 255).astype(np.uint8)

        # 2. Apply the mask to the original BGR image
        # This creates a black background with the object in color
        segmented_cutout = cv2.bitwise_and(image_bgr, image_bgr, mask=binary_mask)

        # Option: If you want a transparent background (PNG), use this instead:
        # b, g, r = cv2.split(image_bgr)
        # rgba = [b, g, r, binary_mask]
        # segmented_cutout = cv2.merge(rgba, 4)

        scale = 0.1  # change as needed (0.25, 0.5, etc.)

        new_width  = int(segmented_cutout.shape[1] * scale)
        new_height = int(segmented_cutout.shape[0] * scale)

        segmented_cutout = cv2.resize(
            segmented_cutout,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # 3. Save to output folder using cv2.imwrite
        output_filename = os.path.join(output_folder, f"segmented_{image_file}")

        # Note: If using the transparent option above, ensure filename ends in .png
        cv2.imwrite(output_filename, segmented_cutout)

        print("\nProcessing complete!")

if __name__ == "__main__":
    main()