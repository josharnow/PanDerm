import cv2
import numpy as np
import albumentations as A
import os
from typing import Literal

## Step 1: Resize Function
def resize_image(image: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    """Resizes an image to the target size. Corresponds to Step 1 in Algorithm 1."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

## Step 2: Normalization Function
def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalizes image pixel values to the [0, 1] range. Corresponds to Step 2."""
    return image.astype(np.float32) / 255.0

def denormalize_image(normalized_image: np.ndarray) -> np.ndarray:
    """Denormalizes image pixel values from [0, 1] back to [0, 255]."""
    return (normalized_image * 255).astype(np.uint8)

## Step 3: Data Augmentation Function
def augment_image(image: np.ndarray, augmentation_type: str) -> np.ndarray:
    """
    Applies a series of data augmentation techniques. Corresponds to Step 3.
    The specific augmentation "functions" are defined inside this step.
    """
    augmentations = []
    # Define the functions for the various augmentation techniques
    match augmentation_type:
        case "rotation":
            augmentations.append(A.Rotate(limit=30, p=1))
        case "flip_horizontal":
            augmentations.append(A.HorizontalFlip(p=1))
        case "flip_vertical":
            augmentations.append(A.VerticalFlip(p=1))
        case "random_cropping":
            augmentations.append(A.RandomResizedCrop(scale=(0.8, 1.0), p=1.0, size=(224, 224))) # NOTE - SkinEHDLF don't specify scale
        case "brightness_adjustment":
            augmentations.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=1)) # NOTE - SkinEHDLF don't specify brightness, contrast, or saturation values

    # NOTE - SkinEHDLF authors apply Gaussean noise only to compare against not applying noise, so it shouldn't be necessary here

    print(augmentations)
    augmentation_pipeline = A.Compose(augmentations)
    
    # Apply the augmentations
    augmented_data = augmentation_pipeline(image=image)
    return augmented_data['image']

## Main Preprocessing Orchestrator
def preprocess_image_stepwise(input_image: np.ndarray, augmentation_type: str | None) -> np.ndarray:
    """
    Runs the full preprocessing pipeline by calling each step's function in order.
    """
    if not augmentation_type:
        # Step 1: Resize
        resized = resize_image(input_image)
        
        # Step 2: Normalize
        result = normalize_image(resized)
    else:
        # Step 3: Augment (using previously normalized image)
        denormalized_image = denormalize_image(input_image) # NOTE - Convert back to 0-255 range for compatibility with augmentation library
        # TODO - Pass normalized input_image to this function when no augmentation is desired; otherwise, pass the raw input_image
        # augmented_image = augment_image(input_image, augmentation_type)
        augmented_image = augment_image(denormalized_image, augmentation_type)
        result = normalize_image(augmented_image)
    return result

## Function to Generate and Save Multiple Augmented Images
def generate_and_save_augmentations(input_path: str, output_dir: str, augmentation_type: str | None):
    """
    Loads a single image and saves multiple, uniquely augmented versions of it
    using the stepwise preprocessing function.
    """
    # Resolve paths relative to this script's directory unless absolute paths are provided
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(input_path):
        input_path = os.path.join(script_dir, input_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    raw_image = cv2.imread(input_path)
    if raw_image is None:
        print(f"Error: Could not read image from {input_path}")
        return

    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    base_filename = os.path.splitext(os.path.basename(input_path))[0]

    # Run the full, stepwise preprocessing pipeline
    processed_image_float = preprocess_image_stepwise(
        input_image=raw_image_rgb,
        augmentation_type=augmentation_type
    )
    
    # Convert image back to a saveable format
    image_to_save = (processed_image_float * 255).astype(np.uint8)
    image_to_save_bgr = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    
    # Create a new filename and save the image
    output_filename = f"{base_filename}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, image_to_save_bgr)
    print(f"Saved to {output_path}")

# --- Example Usage ---
if __name__ == '__main__':
    do_normalization_run = True # Set to True to run normalization

    augmentations_to_do = [
        "rotation", 
        # "flip_horizontal", 
        # "flip_vertical", 
        # "random_cropping", 
        # "brightness_adjustment"
    ]

    # Make default input/output directories one level above this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    if do_normalization_run:
        input_dir = os.path.join(parent_dir, 'raw_images')
        output_dir = os.path.join(parent_dir, 'normalized_images')
        print("Running normalization...")

        for image in os.listdir(input_dir):
            if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(input_dir, image)
            generate_and_save_augmentations(
                input_path=image_path, 
                output_dir=output_dir,
                augmentation_type=None # No augmentation type since we're just normalizing
            )
    

    for augmentation_type in augmentations_to_do:
        print(f"Running augmentation: {augmentation_type}...")
        input_dir = os.path.join(parent_dir, 'normalized_images')
        output_dir = os.path.join(parent_dir, f'image_augmentations/{augmentation_type}')
        for image in os.listdir(input_dir):
            if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(input_dir, image)
            generate_and_save_augmentations(
                input_path=image_path, 
                output_dir=output_dir,
                augmentation_type=augmentation_type
            )