import argparse
from datetime import datetime
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

sys.path.append('./prognosis_model')
from model.additive_model import get_additive_mil_model
from tqdm import tqdm
from dataset.dataset import Dataset, custom_collate_fn, worker_init_fn
from PIL import Image
from PIL import ImageFilter

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from captum.attr import visualization as viz
from captum.attr import Saliency

import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.nn.parallel import DataParallel as parallel

parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--init_model_file', default='Results/prognosis_model/saved_models/model_weights__2024_08_01__13_42_35__55.pth',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--slide_list_filename', default='prognosis_model/data/seg7/test.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--patch_size', default='512', type=int, help='Patch size', dest='patch_size')
parser.add_argument('--num_instances', default='0', type=int, help='number of instances (patches) in a bag', dest='num_instances')
parser.add_argument('--num_bags', default='100', type=int, help='number of instances (patches) in a bag', dest='num_bags')
parser.add_argument('--num_features', default='64', type=int, help='number of features', dest='num_features')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='1', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--attribution_dir', default='Results/prognosis_model/IG_blurred_images', help='Text file to write attribution results', dest='attribution_dir')
parser.add_argument('--cz_imgs_list', default='prognosis_model/data/cz_img_list.txt', help='Image directory', dest='cz_imgs_list')
parser.add_argument('--he_imgs_list', default='prognosis_model/data/he_img_list.txt', help='Image directory', dest='he_imgs_list')
parser.add_argument('--imgs_type', default='HE', help='Image directory', dest='imgs_type')
parser.add_argument('--metrics_dir', default='Results/prognosis_model/test_metrics/', help='Text file to write metrics', dest='metrics_dir')

FLAGS = parser.parse_args()
    
model_name = FLAGS.init_model_file.split('/')[-1][15:-4]


class NormalizationTransform(object):
    def __call__(self, input_tensor):
        mean_tensor = torch.mean(input_tensor).view((1,))
        std_tensor = torch.std(input_tensor).view((1,))

        return TF.normalize(input_tensor, mean_tensor, std_tensor)
        
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
	print(title)
	for i in range(len(feature_names)):
		print(feature_names[i], ": ", '%.3f'%(importances[i]))
	x_pos = (np.arange(len(feature_names)))
	if plot:
		f = plt.figure(figsize=(12,6))
		plt.bar(x_pos, importances, align='center')
		plt.xticks(x_pos, feature_names, wrap=True)
		plt.xlabel(axis_title)
		plt.title(title)

		plt.show()

		return f


img_transforms = transforms.Compose([	
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    NormalizationTransform()
                    ])

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print(device)
# get the model using helper function
model = get_additive_mil_model(num_classes=FLAGS.num_classes, features_size=1024, num_features=FLAGS.num_features, requires_grad= False)

# model = Model(num_classes=FLAGS.num_classes, features_size=1024, num_features=FLAGS.num_features, device_num= 0, requires_grad= True)
# move model to the right device
model.to(device)

if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        state_dict = torch.load(FLAGS.init_model_file, map_location=torch.device(device))
        model.load_state_dict(state_dict['model_state_dict'])

        print("Model weights loaded successfully from file: ", FLAGS.init_model_file)
    else:
        raise Exception("Given model weights file cannot be found!")
else:
    raise Exception("No model weights file is given!")



model = parallel(model, device_ids=[0,1], dim=0)

model.eval()


# read slide list
data_arr = np.loadtxt(FLAGS.slide_list_filename, delimiter='\t', comments='#', dtype=str)
slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]
print('num_slides:{}'.format(num_slides))

# saliency = Saliency(model)
ig = IntegratedGradients(model)

out_dir = '{}/{}/{}_captum'.format(FLAGS.metrics_dir,model_name,FLAGS.slide_list_filename.split('/')[-1][:-4])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with torch.no_grad():

    for s, slide_id in enumerate(slide_ids):
        print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

        out_path = '{}/{}'.format(out_dir,slide_id)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        slide_label = labels[s]

        # dataset for the current slide
        dataset = Dataset(slide_list_filename= FLAGS.slide_list_filename, slide_id=slide_id, cz_imgs_list= FLAGS.cz_imgs_list, he_imgs_list= FLAGS.he_imgs_list, imgs_type= FLAGS.imgs_type)

        # define data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)

           
        for images, targets, cz_img_paths, he_img_paths in data_loader:
            
            inputs = images
            images = images.to(device)
            targets = targets.to(device)

            num_instances = images.shape[0]
            images.requires_grad_()
            
            
            attr_arr_sum = np.zeros((num_instances, 224,224,3),dtype=np.float32) # height, width, channels
            for k in range(10):
                baseline_tensor = [] #torch.randint(0, num_instances, (num_instances,3,224,224))

                baseline_tensor = torch.empty((0, 3, 224, 224), dtype=torch.float32)

                for instance in range(num_instances):
                    if FLAGS.imgs_type == 'HE':
                        original_image = Image.open(he_img_paths[0][instance]).convert('RGB') # width, height, channels
                        baseline_instance = img_transforms(original_image.filter(ImageFilter.BoxBlur(radius=15))).unsqueeze(0)
                        baseline_tensor = torch.cat((baseline_tensor, baseline_instance), dim=0)
                    else:
                        original_image = Image.open(cz_img_paths[0][instance]).convert('RGB') # width, height, channels
                        baseline_instance = img_transforms(original_image.filter(ImageFilter.BoxBlur(radius=15))).unsqueeze(0)
                        baseline_tensor = torch.cat((baseline_tensor, baseline_instance), dim=0)

                # mu=torch.mean(baseline_tensor*1.,dim=(1,2,3),keepdim=True)
                # std=torch.std(baseline_tensor*1.,dim=(1,2,3),keepdim=True)
                # baseline_tensor = (baseline_tensor - mu) / std

                attr = ig.attribute(images, baselines= baseline_tensor.cuda(images.get_device()), target=targets, n_steps=10, return_convergence_delta=False) 

                attr = attr.cpu().detach().numpy()
                attr = np.transpose(attr, (0,2,3,1))  # batch_size, width, height, channels

                attr_arr = np.zeros((num_instances,224,224,3),dtype=np.float32)

                attr_arr = attr[:,:,:,:]
                
                attr_arr_sum += attr_arr

            attr_arr_mean = attr_arr_sum/10

            for instance in range(num_instances):
                if FLAGS.imgs_type == 'HE':
                    original_image = Image.open(he_img_paths[0][instance]).convert('RGB') # width, height, channels
                else:
                    original_image = Image.open(cz_img_paths[0][instance]).convert('RGB') # width, height, channels
                
                img_arr = np.array(original_image)

                attr_arr_mean_instance = attr_arr_mean[instance]

                np.save('{}/{}__img.npy'.format(out_path, instance),img_arr)
                np.save('{}/{}__attribution.npy'.format(out_path, instance),attr_arr_mean_instance)