import cv2
import os

# List of files to process
image_files = ["sach.jpg", "segmented/segmented_sach.jpg"]

for filename in image_files:
    # Check if the file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        continue

    # Read the image
    img = cv2.imread(filename)

    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Could not read image {filename}.")
        continue

    # Flip the image around its y-axis (horizontal flip)
    # flipCode 1 specifies a horizontal flip
    flipped_img = cv2.flip(img, 1)

    # Generate the output filename
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_flipped{ext}"

    # Save the flipped image
    cv2.imwrite(output_filename, flipped_img)
    print(f"Successfully flipped {filename} and saved as {output_filename}")