import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image



class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir=None, img_ext= '', crops_info=None, slide_id=None, transforms=None):
        self._image_dir = image_dir
        self._img_ext = img_ext
        self._crops_info = crops_info
        self._transforms = transforms
        self._slide_id = slide_id
        self._cropped = []
        self._slide_num_patches = self.get_slide_info()[0]
        # print(self._slide_num_patches)


    @property
    def num_patches(self):
        return self._slide_num_patches


    def __len__(self):
        return self._slide_num_patches

    def get_slide_info(self):
        # read cropped patches info file
        dataset_info = np.loadtxt(self._crops_info, delimiter='\t', comments='#', dtype=str)
        slide_ids = dataset_info[:,1]
        num_patches = np.asarray(dataset_info[:,0], dtype=int)

        return num_patches[slide_ids == self._slide_id]

    def get_idx(self, temp_img_dir, idx):
        try:
                
            img_name = self._slide_id + '-' + str(idx)
    
            img_path = '{}/{}/{}.{}'.format(temp_img_dir, self._slide_id, img_name, self._img_ext)

            if not os.path.exists(img_path) or idx in self._cropped:
                return self.get_idx(temp_img_dir, idx+1)          
            else: 
                return img_path
            
        except Exception as ex:
            raise Exception(ex)

    def get_image(self, img_dir, patch_id):
        
        # img_path = '{}/{}.jpeg'.format(img_dir, patch_id)
        img_path = self.get_idx(self._image_dir, patch_id)

        self._cropped.append(patch_id)

        img = Image.open(img_path).convert("RGB")

        return self._transforms(img)

        
    def __getitem__(self, idx):

        temp_img_dir = os.path.join(self._image_dir, self._slide_id)

        image = self.get_image(img_dir = temp_img_dir, patch_id= idx)

        return image




