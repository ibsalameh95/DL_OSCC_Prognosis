import os
from PIL import Image

def crop_and_save_images(base_dir, output_dir, crop_size=(512, 512)):
    overlap = 224
    # Loop through each side folder (e.g., SIDES)
    for slide in os.listdir(base_dir):
        slide_path = os.path.join(base_dir, slide)
        if os.path.isdir(slide_path):

            output_slide_path = os.path.join(output_dir, slide)
            os.makedirs(output_slide_path, exist_ok=True)
            
            # Loop through each image in the slide folder
            for image_name in os.listdir(slide_path):
                image_path = os.path.join(slide_path, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    image = Image.open(image_path)
                    width, height = image.size
                    
                    crop_id = 1
                    step_x = crop_size[0] - overlap
                    step_y = crop_size[1] - overlap
                    
                    # Loop to create overlapping crops
                    for top in range(0, height - crop_size[1] + 1, step_y):
                        for left in range(0, width - crop_size[0] + 1, step_x):
                            # Define the cropping box (left, top, right, bottom)
                            right = left + crop_size[0]
                            bottom = top + crop_size[1]
                            
                            # Crop and save the image
                            crop = image.crop((left, top, right, bottom))
                            crop_filename = f"{os.path.splitext(image_name)[0]}_{crop_id}.png"
                            crop.save(os.path.join(output_slide_path, crop_filename))
                            
                            crop_id += 1

if __name__ == "__main__":
    base_dir = 'Data/CroppedRegions/WholeImage/Confocal'
    output_dir = 'Data/CroppedRegions/WholeImageCrops/Confocal'
    
    crop_and_save_images(base_dir, output_dir)
