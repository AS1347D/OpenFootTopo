import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

# ----- folders -----
INPUT_DIR = "GeomDir/segmented"
COORDS_DIR = "GeomDir/coordinates"
COORDSPLOT_DIR = "GeomDir/coordinatesPlots"
EDGES_DIR = "GeomDir/edges_overlay"

# Create directories once at the start
os.makedirs(COORDS_DIR, exist_ok=True)
os.makedirs(EDGES_DIR, exist_ok=True)
os.makedirs(COORDSPLOT_DIR, exist_ok=True)

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ---------- helpers ----------
def build_mask(img_bgr):
    """
    Build a clean binary mask for a light object on a black/dark background.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur to smooth noise and reduce jagged edges before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Thresholding
    # Since background is black, any pixel significantly > 0 is foreground.
    # Otsu is robust, but a fixed low threshold (e.g., 10-15) is often safer for black backgrounds.
    _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Morphological Cleanup (fills small holes, removes isolated noise)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask

def largest_component(mask):
    """Keep only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask
    
    # stats[:, 4] is area. Index 0 is background, so start argmax from index 1
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ((labels == largest_label).astype(np.uint8) * 255)

def extract_edges_and_contour(mask):
    """
    Return edge pixels (1-px boundary) using contours for precision.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros_like(mask), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))

    # Get the largest contour (the foot)
    c = max(contours, key=cv2.contourArea)
    
    # 1. Create a clean 1-pixel wide edge map directly from the contour
    # This is often cleaner than morphological gradient
    edge_map = np.zeros_like(mask)
    cv2.drawContours(edge_map, [c], -1, 255, 1) # Thickness 1
    
    # 2. Extract (x,y) coordinates from the edge map
    ys, xs = np.where(edge_map > 0)
    edge_pixels = np.column_stack([xs, ys])

    # 3. Contour coordinates (ordered)
    contour_coords = c.reshape(-1, 2)
    
    # 4. Approx coordinates (simplified polygon)
    eps = 0.001 * cv2.arcLength(c, True) # Smaller epsilon for more detail

    return edge_map, edge_pixels, contour_coords

def save_outputs(img_bgr, base, edge_map, edge_pixels, contour_coords):

    # Calculate centers for both axes
    y_center = (contour_coords[:, 1].min() + contour_coords[:, 1].max()) / 2

    # Mirror across both horizontal (y_center) and vertical (x_center) axes
    mirrored_coords = contour_coords.copy()
    mirrored_coords[:, 1] = 2 * y_center - contour_coords[:, 1]

    # Clean up (remove first point if it was a duplicate/closed loop artifact)
    mirrored_coords = np.delete(mirrored_coords, 0, axis=0)

    plot_coords = mirrored_coords.copy()

    # # translate to zero
    min_x = np.min(mirrored_coords[:, 0])
    mirrored_coords[:, 0] = mirrored_coords[:, 0] - min_x
    min_y = np.min(mirrored_coords[:, 1])
    mirrored_coords[:, 1] = mirrored_coords[:, 1] - min_y

    # scale for real dimensions
    # mirrored_coords: shape (N, 2) -> columns [x, y] in pixels
    x = mirrored_coords[:, 0]
    x_min, x_max = x.min(), x.max()
    x_range_px = x_max - x_min

    target_x_mm = 260.0
    mm_per_px = target_x_mm / x_range_px   # scale factor

    # uniform scaling (keeps aspect ratio)
    coords_mm = mirrored_coords.copy().astype(float)
    coords_mm[:, 0] = mirrored_coords[:, 0] * mm_per_px
    coords_mm[:, 1] = mirrored_coords[:, 1] * mm_per_px
    
    # save CSV of edge pixels
    pd.DataFrame(coords_mm, columns=["x", "y"]).to_csv(
        os.path.join(COORDS_DIR, f"{base}_edge_pixels.csv"), index=False
    )

    # overlay: edge map in green
    # Draw strictly on top of the original image
    overlay_edges = img_bgr.copy()
    
    # Make edges bright Green (0, 255, 0)
    # Using the edge_map mask to apply color
    overlay_edges[edge_map > 0] = [0, 255, 0]
    
    cv2.imwrite(os.path.join(EDGES_DIR, f"{base}_edge_overlay.png"), overlay_edges)

    # optional: scatter plot of edge pixels
    if edge_pixels.size:
        h, w = img_bgr.shape[:2]
        fig_w = 6
        fig_h = fig_w * (h / w)

        plt.figure(figsize=(fig_w, fig_h))
        # Plot ordered contour points for a continuous line look, 
        # or scatter edge_pixels for raw dots
        plt.plot(plot_coords[:, 0], plot_coords[:, 1], c='red', linewidth=2)
        
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0, w)
        plt.ylim(0, h)
        # plt.title(f"{base} (Contour)")
        plt.xlabel(r"$p_x$ (pixels)")
        plt.ylabel(r"$p_y$ (pixels)")
        # plt.axis('off') # Cleaner look
        plt.tight_layout()
        plt.savefig(os.path.join(COORDSPLOT_DIR, f"{base}_edge_points.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

def process_one(path):
    base = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Skipping unreadable file: {path}")
        return
        
    # Build mask
    mask = build_mask(img)
    mask = largest_component(mask)
    
    # Extract edges
    edge_map, edge_pixels, contour_coords = extract_edges_and_contour(mask)
    
    # Save
    save_outputs(img, base, edge_map, edge_pixels, contour_coords)
    print(f"Processed {base}: {len(edge_pixels)} edge pixels found.")

# ---------- run over the folder ----------
files = []
for ext in VALID_EXTS:
    # Handle both *.ext and *uppercase.EXT if on Linux/Case-sensitive
    files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
    files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))

# Remove duplicates if any (e.g. from case sensitivity overlap)
files = sorted(list(set(files)))

if not files:
    print(f"No images found in '{INPUT_DIR}'.")
else:
    print(f"Found {len(files)} images. Processing...")
    for f in files:
        process_one(f)
    print("\nDone.")
    print(f"Coordinates → {COORDS_DIR}")
    print(f"Overlays → {EDGES_DIR}")