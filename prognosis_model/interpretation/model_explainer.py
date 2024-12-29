import argparse
from datetime import datetime
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# from model import Model


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb

sys.path.append('./prognosis_model')
from model.additive_model import get_additive_mil_model
from tqdm import tqdm
from dataset.dataset import Dataset, custom_collate_fn, worker_init_fn

parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--init_model_file', default='Results/prognosis_model/saved_models/model_weights__2024_07_12__20_37_30__39.pth',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--slide_list_filename', default='prognosis_model/data/seg6/test_selected.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--patch_size', default='512', type=int, help='Patch size', dest='patch_size')
parser.add_argument('--num_instances', default='0', type=int, help='number of instances (patches) in a bag', dest='num_instances')
parser.add_argument('--num_bags', default='100', type=int, help='number of instances (patches) in a bag', dest='num_bags')
parser.add_argument('--num_features', default='256', type=int, help='number of features', dest='num_features')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='1', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--metrics_dir', default='Results/prognosis_model/test_metrics/', help='Text file to write metrics', dest='metrics_dir')
parser.add_argument('--cz_imgs_list', default='prognosis_model/data/cz_img_list.txt', help='Image directory', dest='cz_imgs_list')
parser.add_argument('--he_imgs_list', default='prognosis_model/data/he_img_list.txt', help='Image directory', dest='he_imgs_list')
parser.add_argument('--seg_imgs_list', default='prognosis_model/Data/seg_img_list.txt', help='Image directory', dest='seg_imgs_list')
parser.add_argument('--imgs_type', default='HE', help='Image directory', dest='imgs_type')

FLAGS = parser.parse_args()
    
model_name = FLAGS.init_model_file.split('/')[-1][15:-4]

out_dir = '{}/{}/{}_att_values_selected_mask'.format(FLAGS.metrics_dir,model_name,FLAGS.slide_list_filename.split('/')[-1][:-4])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('init_model_file: {}'.format(FLAGS.init_model_file))
print('slide_list_filename: {}'.format(FLAGS.slide_list_filename))
print('patch_size: {}'.format(FLAGS.patch_size))
print('imgs_type: {}'.format(FLAGS.imgs_type))
print('num_instances: {}'.format(FLAGS.num_instances))
print('num_bags: {}'.format(FLAGS.num_bags))
print('num_features: {}'.format(FLAGS.num_features))
print('num_classes: {}'.format(FLAGS.num_classes))
print('batch_size: {}'.format(FLAGS.batch_size))
print('metrics_dir: {}'.format(FLAGS.metrics_dir))



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# get the model using helper function
# model = Model(num_classes=FLAGS.num_classes, features_size=1024, num_features=FLAGS.num_features, requires_grad= False)
model = get_additive_mil_model(num_classes=FLAGS.num_classes, features_size=1024, num_features=FLAGS.num_features, requires_grad= False)

# move model to the right device
model.to(device)

def classifier_fn(images):
    model.eval()
    with torch.no_grad():
        # Convert images to torch tensor
        # images = torch.from_numpy(images[0]).permute(0, 3, 1, 2).float().to(device) # (N, H, W, C)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device) # (N, H, W, C)
        
        outputs = model(images)
        patch_logits = outputs['patch_logits']
        predictions = outputs['value']
        # Convert outputs to numpy array

        # probs = F.softmax(patch_logits, dim=2).cpu().numpy()
        probs = F.softmax(predictions, dim=1).cpu().numpy()
        # input('ds')
    return probs

def tensor_to_numpy(img_tensor):
    img_array = img_tensor.cpu().numpy()
    img_array = np.transpose(img_array, (0, 2, 3, 1))  # Convert from (N, C, H, W) to (N, H, W, C)
    return img_array


def load_torch_model(model, weights):
    state_dict = torch.load(weights, map_location=torch.device(device))
    state_dict = state_dict['model_state_dict']
    # print(state_dict['model_state_dict'])
    # input('s')
    print(model.load_state_dict(state_dict))
    print("Model loading complete ...")

if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        
        load_torch_model(model, FLAGS.init_model_file)
        print("Model weights loaded successfully from file: ", FLAGS.init_model_file)
    else:
        raise Exception("Given model weights file cannot be found!")
else:
    raise Exception("No model weights file is given!")


# read slide list
data_arr = np.loadtxt(FLAGS.slide_list_filename, delimiter='\t', comments='#', dtype=str)

patients_mapping = np.loadtxt('Data/patients_mapping.txt', delimiter='\t', comments='#', dtype=str) 

slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]

print('num_slides:{}'.format(num_slides))

explainer = lime_image.LimeImageExplainer()

model.eval()

colors=['#DDDDDD', '#3E4CB9']

with torch.no_grad():

    for s, slide_id in enumerate(slide_ids):
        print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

        slide_label = labels[s]

        # dataset for the current slide
        dataset = Dataset(slide_list_filename= FLAGS.slide_list_filename, slide_id=slide_id, cz_imgs_list= FLAGS.cz_imgs_list, he_imgs_list= FLAGS.he_imgs_list, imgs_type= FLAGS.imgs_type)

        # define data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)


        torch.set_printoptions(profile="full")
        bag_count = 0
        pbar = tqdm(total=len(data_loader))
        for images, lable, cz_img_paths, he_img_paths in data_loader:
            # print(img.shape) [[0.000017 0.999983]]
            
            img = images.to(device)
            # get logits from the model
            output = model(img)
            
            predictions = output['value']
            patch_logits = output['patch_logits']
            # Convert outputs to numpy array

            probs = F.softmax(predictions, dim=1).cpu().numpy()

            bag = tensor_to_numpy(images) # (N, H, W, C)

            num_instances = img.shape[0]

            explanation = explainer.explain_instance(
                bag,
                classifier_fn = classifier_fn,  # Use the classifier function
                top_labels= 1,
                hide_color=None,
                batch_size= num_instances,
                num_features= 256,
                num_samples= 300  # Increase num_samples for better diversity in explanations
            )   
            # Get the image and mask for the top label

            temp_p, mask_p = explanation.get_image_and_mask(
                label= explanation.top_labels[0],
                num_features=256,
                positive_only = True,
                hide_rest=False
            )

            temp_all, mask_all = explanation.get_image_and_mask(
                label= explanation.top_labels[0],
                num_features=256,
                positive_only = False,
                negative_only = False,
                hide_rest=False
            )

            temp_n, mask_n = explanation.get_image_and_mask(
                label= explanation.top_labels[0],
                num_features=256,
                positive_only = False,
                negative_only = True,
                hide_rest=False
            )

            attention_values = output['patch_logits']

            attention_values_probs = F.softmax(attention_values, dim=2)

            output = output['value']

            # obtain probs
            probs = F.softmax(output, dim=1)

            # obtain predictions
            _, predicted = torch.max(output, 1)
            
            predicted_arr = predicted.cpu().numpy()
            probs_arr = probs.cpu().numpy()

            temp_num_predictions = predicted_arr.shape[0]

            image_size = (224,224)

            out_path = '{}/{}'.format(out_dir, slide_id)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                
            if num_instances <= 2:
                continue

            for instance in range(num_instances):
                image_index = instance
                # Create the plot for each page
                fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

                if FLAGS.imgs_type == 'CZ_HE':
                    
                    if image_index < len(cz_img_paths[0]):
                        original_image = Image.open(cz_img_paths[0][image_index]).convert('RGB') # width, height, channels
                        original_image = np.array(original_image.resize(image_size))

                    if image_index >= len(cz_img_paths[0]):
                        original_image = Image.open(he_img_paths[0][image_index-num_instances]).convert('RGB') # width, height, channels
                        original_image = np.array(original_image.resize(image_size))

                if FLAGS.imgs_type == 'CZ':
                    original_image = Image.open(cz_img_paths[0][image_index]).convert('RGB') # width, height, channels
                    original_image = np.array(original_image.resize(image_size))

                if FLAGS.imgs_type == 'HE':
                    original_image = Image.open(he_img_paths[0][image_index]).convert('RGB') # width, height, channels
                    original_image = np.array(original_image.resize(image_size))

                # overlay = label2rgb(mask_combined, image=original_image / 255.0, colors=['red', 'blue'], bg_label=0)

                probs_att = attention_values_probs.squeeze(0).cpu()[image_index]

                att_values = attention_values.squeeze(0).cpu()[image_index]

                ax_img = axes[0]
                ax_img.imshow(original_image / 255.0, cmap='gray')
                ax_img.axis('off')
                ax_img.set_title(f'Image {image_index + 1}')

                ax_img_0 = axes[1]
                ax_img_0.imshow(mask_all[image_index], cmap='gray')
                ax_img_0.axis('off')
                ax_img_0.set_title(f'Full mask')


                ax_img_1 = axes[2]
                ax_img_1.imshow(mask_p[image_index], cmap='gray')
                ax_img_1.axis('off')
                ax_img_1.set_title(f'Postive mask')

                ax_img_2 = axes[3]
                ax_img_2.imshow(mask_n[image_index], cmap='gray')
                ax_img_2.axis('off')
                ax_img_2.set_title(f'Negative mask')

                # ax_img_1 = axes[2]
                # ax_img_1.imshow(mark_boundaries(original_image / 255.0, mask[image_index], color=(1, 0, 0)), cmap='gray')
                # ax_img_1.axis('off')
                # ax_img_1.set_title(f'Positive Attribution')

                # Plot the bar chart
                # ax_bar = axes[3]
                # ax_bar.bar(['Good Prog', 'Poor Prog'], [att_values[0].item(), att_values[1].item()], color=colors) 

                # ax_bar.set_xlim(0, 1)
                # ax_bar.set_ylim(0, 1)

                # # Remove the right and top borders
                # ax_bar.spines['right'].set_visible(False)
                # ax_bar.spines['top'].set_visible(False)               

                # ax_bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                # ax_bar.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)  # Enable y-axis labels
                # ax_bar.set_title(f'0: {att_values[0].item():.3f}, 1: {att_values[1].item():.3f}')
                

                plt.tight_layout()
                plt.savefig('{}/{}.png'.format(out_path, instance+1), dpi=300)
                fig.clear()
                # input('ds')

        pbar.update(1)

        pbar.close()
