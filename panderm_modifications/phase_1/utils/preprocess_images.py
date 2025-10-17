import cv2
import numpy as np
import albumentations as A
import os
import glob

## Step 1: Resize Function
def resize_image(image: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    """Resizes an image to the target size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

## Step 2: Normalization Function
def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalizes image pixel values from [0, 255] to the [0, 1] range."""
    return image.astype(np.float32) / 255.0

## Step 3: Data Augmentation Function
def get_augmentation_pipeline(augmentation_type: str) -> A.Compose:
    """Returns the appropriate augmentation pipeline for a given type."""
    augmentations = []
    match augmentation_type:
        case "rotation":
            augmentations.append(A.Rotate(limit=30, p=1.0))
        case "flip_horizontal":
            augmentations.append(A.HorizontalFlip(p=1.0))
        case "flip_vertical":
            augmentations.append(A.VerticalFlip(p=1.0))
        case "random_cropping":
            augmentations.append(A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0)) # NOTE - SkinEHDLF don't specify scale
        case "brightness_adjustment":
            augmentations.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0)) # NOTE - SkinEHDLF don't specify brightness, contrast, or saturation values
    
    # NOTE - SkinEHDLF authors apply Gaussean noise only to compare against not applying noise, so it shouldn't be necessary here
    return A.Compose(augmentations)

## Main Orchestrator and Saving Function
def process_and_save_image(input_path: str, output_dir: str, augmentation_type: str | None):
    """
    Loads a raw image, applies the full, correct processing pipeline, and saves the result.
    """
    # 1. Load Image (Result is BGR)
    raw_image = cv2.imread(input_path)
    if raw_image is None:
        print(f"Warning: Could not read image from {input_path}, skipping.")
        return

    # 2. Convert to RGB for processing
    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # 3. Apply transformations in the correct order
    if augmentation_type == "random_cropping":
        # This function handles its own resizing. Apply to raw image first.
        pipeline = get_augmentation_pipeline(augmentation_type)
        augmented_image = pipeline(image=raw_image_rgb)['image']
        # Then normalize.
        final_processed_image = normalize_image(augmented_image)
    elif augmentation_type:
        # For other augmentations, resize the image first.
        resized_image = resize_image(raw_image_rgb)
        pipeline = get_augmentation_pipeline(augmentation_type)
        # Then apply the augmentation to the resized image.
        augmented_image = pipeline(image=resized_image)['image']
        # Finally, normalize.
        final_processed_image = normalize_image(augmented_image)
    else: # This handles the case for normalization only
        resized_image = resize_image(raw_image_rgb)
        final_processed_image = normalize_image(resized_image)

    # 4. Convert back to a saveable format
    image_to_save = (final_processed_image * 255).astype(np.uint8)
    image_to_save_bgr = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    
    # 5. Save the final image
    base_filename = os.path.basename(input_path)
    base_filename_no_ext = os.path.splitext(base_filename)[0] # Get the base filename WITHOUT the original extension
    output_filename = f"{base_filename_no_ext}.png" # Create a new filename with the .png extension
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, image_to_save_bgr)

# --- Example Usage ---
if __name__ == '__main__':
    augmentations_to_do = [
        "rotation", 
        "flip_horizontal", 
        "flip_vertical", 
        "random_cropping", 
        "brightness_adjustment"
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_dir = os.path.dirname(script_dir)
    parent_dir = os.path.dirname(script_dir)
    input_dir = os.path.join(parent_dir, 'raw_images')
    output_base_dir = os.path.join(parent_dir, 'processed_images')

    # Get a list of all image paths from the input directory
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    if not image_paths:
        print(f"Error: No images found in the '{input_dir}' directory. Please add some images to process.")
        exit(1)

    # Run Normalization Only
    print("--- Running Normalization Only ---")
    norm_output_dir = os.path.join(output_base_dir, 'normalized_images') # NOTE - THIS FOLDER CONTAINS IMAGES THAT HAVE ONLY BEEN NORMALIZED FROM RAW
    for path in image_paths:
        process_and_save_image(path, norm_output_dir, augmentation_type=None)
    print("Normalization complete.")

    # Run Augmentations
    for aug_type in augmentations_to_do:
        print(f"--- Running Augmentation: {aug_type} ---")
        aug_output_dir = os.path.join(output_base_dir, f'image_augmentations/{aug_type}')
        for path in image_paths:
            # THIS IS THE KEY FIX: Pass the full 'path' to the function
            process_and_save_image(path, aug_output_dir, augmentation_type=aug_type)
        print(f"Augmentation '{aug_type}' complete.")