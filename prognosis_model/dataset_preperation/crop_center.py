import os
import numpy as np
from PIL import Image
import imageio



def crop_Center():
    path = 'Data/CroppedRegions/1024/BrightfieldSegmented'

    crop_width = 224
    crop_height = 224

    for root, dnames, fnames in os.walk(path):
        for f in fnames:
            if f.endswith('.png'):
                img_path = os.path.join(root, f)
                img_arr = np.array(Image.open(img_path))

                img_height, img_width = img_arr.shape[:2]
                
                if img_width >= crop_width and img_height >= crop_height:

                    start_x = max(img_width//2 - (crop_width//2), 0)
                    start_y = max(img_height//2 - (crop_height//2), 0)
                    end_x = min(start_x + crop_width, img_width)
                    end_y = min(start_y + crop_height, img_height)
                    
                    cropped_img = img_arr[start_y:end_y, start_x:end_x]
                    
                    out_dir = root.replace('1024', '224')
                    # out_dir = out_dir.replace('Level0/', '')

                    # out_dir = 'Data/CroppedRegions/1024/Confocal'
                    
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    out_path = os.path.join(out_dir, f)
                
                    imageio.imwrite(out_path, cropped_img)

crop_Center()