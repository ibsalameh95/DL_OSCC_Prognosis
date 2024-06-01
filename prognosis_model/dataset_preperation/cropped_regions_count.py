import os
import glob

def count_images_in_directories(root_dir):
    with open("Data/CroppedRegions/cropped_patches_info.txt", "w") as output_file:
        for dir_name, _, files in os.walk(root_dir):
            image_count = len(glob.glob(os.path.join(dir_name, "*.png")))  # Add more extensions if needed
            if image_count > 0:
                output_file.write(f"{image_count}\t{os.path.basename(dir_name)}\n")

# Example usage:
root_directory = "Data/CroppedRegions/512/Brightfield"

count_images_in_directories(root_directory)
