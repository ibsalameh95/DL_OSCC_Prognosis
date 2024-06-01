import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
import torchvision


class Dataset(torch.utils.data.Dataset):
    def __init__(self, slide_id=None, slide_list_filename=None, cz_imgs_list= None, he_imgs_list= None, seg_imgs_list= None, num_instances=16, dataset_type='train', imgs_type= None):
        self._dataset_type = dataset_type
        self._slide_id = slide_id
        #transforms.ToTensor() #torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms() #
        self._cz_transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms() # self.cz_image_transforms()  # torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms() #
        self._he_transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms() # self.he_image_transforms()  # torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms() #       
        self._num_instances = num_instances
        self._imgs_type = imgs_type

        self._slide_ids, self._slide_labels = self.read_slide_list(slide_list_filename)

        self._cz_imgs_list = np.loadtxt(cz_imgs_list, delimiter='\t', comments='#', dtype=str)
        self._he_imgs_list = np.loadtxt(he_imgs_list, delimiter='\t', comments='#', dtype=str)
        self._seg_imgs_list = np.loadtxt(seg_imgs_list, delimiter='\t', comments='#', dtype=str)

        self._num_slides = len(self._slide_ids)

    @property
    def num_slides(self):
        return self._num_slides


    def __len__(self):
        return self._num_slides

    def cz_image_transforms(self):
        if self._dataset_type == 'train':

            img_transforms = transforms.Compose([	
                                transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                self.NormalizationTransform()
                                ])
        else: 
            img_transforms = transforms.Compose([	
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                self.NormalizationTransform()
                                ])            

        return img_transforms

    def he_image_transforms(self):
        if self._dataset_type == 'train':

            img_transforms = transforms.Compose([	
                                transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                self.NormalizationTransform()
                                ])
        else: 
            img_transforms = transforms.Compose([	
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                self.NormalizationTransform()
                                ])            

        return img_transforms
    
    class NormalizationTransform(object):
        def __call__(self, input_tensor):
            mean_tensor = torch.mean(input_tensor).view((1,))
            std_tensor = torch.std(input_tensor).view((1,))

            return TF.normalize(input_tensor, mean_tensor, std_tensor)
        
    def read_slide_list(self, slide_list_filename):
        data_arr = np.loadtxt(slide_list_filename, delimiter='\t', comments='#', dtype=str)
        slide_ids = data_arr[:,0]
        labels = np.asarray(data_arr[:,1], dtype=int)

        slide_ids_list = []
        slide_num_patches_list = []
        slide_labels_list = []
        for i,temp_slide_id in enumerate(slide_ids):
            temp_label = labels[i]

            slide_ids_list.append(temp_slide_id)
            slide_labels_list.append(temp_label)

        return slide_ids_list, slide_labels_list

        
    def construct_slide_bag(self, slide_id):

        idx = self._slide_ids.index(slide_id)
        slide_label = self._slide_labels[idx]

        slide_images = torch.empty((0, 3, 224, 224), dtype=torch.float32)
        
        cz_img_paths = self._cz_imgs_list[self._cz_imgs_list[:, 0] == slide_id][:,3]
        he_img_paths = self._he_imgs_list[self._he_imgs_list[:, 0] == slide_id][:,3]  
        seg_img_paths = self._seg_imgs_list[self._seg_imgs_list[:, 0] == slide_id][:,3]  


        if self._imgs_type == 'CZ':
                for i, path in enumerate(cz_img_paths):
                    img = self._cz_transforms(Image.open(path).convert("RGB")).unsqueeze(0)
                    slide_images = torch.cat((slide_images, img), dim=0)

        if self._imgs_type == 'HE':
                for i, path in enumerate(he_img_paths):
                    img = self._he_transforms(Image.open(path).convert("RGB")).unsqueeze(0)
                    slide_images = torch.cat((slide_images, img), dim=0)

        if self._imgs_type == 'CZ_HE':

                for i, path in enumerate(cz_img_paths):
                    img = self._cz_transforms(Image.open(path).convert("RGB")).unsqueeze(0)
                    slide_images = torch.cat((slide_images, img), dim=0)

                for i, path in enumerate(he_img_paths):
                    img = self._he_transforms(Image.open(path).convert("RGB")).unsqueeze(0)
                    slide_images = torch.cat((slide_images, img), dim=0)

        if self._imgs_type == 'CZ_HE_SEG':

                for i, path in enumerate(cz_img_paths):
                    img = self._cz_transforms(Image.open(path).convert("RGB")).unsqueeze(0)
                    slide_images = torch.cat((slide_images, img), dim=0)

                for i, path in enumerate(he_img_paths):
                    img = self._he_transforms(Image.open(path).convert("RGB")).unsqueeze(0)
                    slide_images = torch.cat((slide_images, img), dim=0)

                for i, path in enumerate(seg_img_paths):
                    img = self._he_transforms(Image.open(path).convert("RGB")).unsqueeze(0)
                    slide_images = torch.cat((slide_images, img), dim=0)
                    
        return slide_images, slide_label
        

    def __getitem__(self, idx):

        temp_slide_id = self._slide_ids[idx]
        temp_slide_label = self._slide_labels[idx]

        slide_images, slide_label = self.construct_slide_bag(slide_id = temp_slide_id)
        
        temp_slide_label = torch.as_tensor(temp_slide_label, dtype=torch.int64)

        return slide_images, temp_slide_label, temp_slide_id

def custom_collate_fn(batch):
    images_list, slide_label, temp_slide_id = zip(*batch)

    return torch.cat(images_list,dim=0), torch.stack(slide_label,dim=0), temp_slide_id

def worker_init_fn(id):
    np.random.seed(torch.initial_seed()&0xffffffff)