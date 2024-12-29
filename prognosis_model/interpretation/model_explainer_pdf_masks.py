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

from skimage.color import label2rgb
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

from matplotlib.colors import Normalize

parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--init_model_file', default='Results/prognosis_model/saved_models/model_weights__2024_08_16__17_49_30__35.pth',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--slide_list_filename', default='prognosis_model/data/seg12/test.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--patch_size', default='512', type=int, help='Patch size', dest='patch_size')
parser.add_argument('--num_instances', default='0', type=int, help='number of instances (patches) in a bag', dest='num_instances')
parser.add_argument('--num_bags', default='100', type=int, help='number of instances (patches) in a bag', dest='num_bags')
parser.add_argument('--num_features', default='64', type=int, help='number of features', dest='num_features')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='1', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--metrics_dir', default='Results/prognosis_model/test_metrics/', help='Text file to write metrics', dest='metrics_dir')
parser.add_argument('--cz_imgs_list', default='prognosis_model/data/cz_img_list.txt', help='Image directory', dest='cz_imgs_list')
parser.add_argument('--he_imgs_list', default='prognosis_model/data/he_img_list.txt', help='Image directory', dest='he_imgs_list')
parser.add_argument('--seg_imgs_list', default='prognosis_model/Data/seg_img_list.txt', help='Image directory', dest='seg_imgs_list')
parser.add_argument('--imgs_type', default='CZ_HE', help='Image directory', dest='imgs_type')

FLAGS = parser.parse_args()
    
model_name = FLAGS.init_model_file.split('/')[-1][15:-4]

out_dir = '{}/{}/{}_heatmap'.format(FLAGS.metrics_dir,model_name,FLAGS.slide_list_filename.split('/')[-1][:-4])
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
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device) # (N, H, W, C)
        
        outputs = model(images)
        patch_logits = outputs['patch_logits']
        predictions = outputs['value']
        # Convert outputs to numpy array

        probs = F.softmax(predictions, dim=1).cpu().numpy()

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

            num_instances = img.shape[0]
            # get logits from the model
            output = model(img)
            
            predictions = output['value']
            patch_logits = output['patch_logits']
            # Convert outputs to numpy array

            probs = F.softmax(predictions, dim=1).cpu().numpy()

            bag = tensor_to_numpy(images) # (N, H, W, C)
        
            explanation = explainer.explain_instance(
                bag,
                classifier_fn = classifier_fn,  # Use the classifier function
                top_labels= 1,
                hide_color=None,
                batch_size= num_instances,
                num_features= 64,
                num_samples=1000  # Increase num_samples for better diversity in explanations
            )   
            # Get the image and mask for the top label

            temp_p, mask_p = explanation.get_image_and_mask(
                label= explanation.top_labels[0],
                num_features=64,
                positive_only = True,
                hide_rest=True
            )

            temp_n, mask_n = explanation.get_image_and_mask(
                label= explanation.top_labels[0],
                num_features=64,
                positive_only = False,
                negative_only = True,
                hide_rest=True
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

            images_per_row = 1  # Number of images per row
            num_cols_per_image = 4  # Each image and bar plot pair
            num_cols = images_per_row * num_cols_per_image  # Total columns per row
            image_size = (224, 224)

            # Define the number of images per row and per page
            images_per_page = 6
            num_pages = num_instances // images_per_page + int(num_instances % images_per_page > 0)

            pdf_pages = PdfPages('{}/{}.pdf'.format(out_dir, slide_id))

            # Calculate the number of rows required
            num_rows = (num_instances + images_per_row - 1) // images_per_row

            ind =  explanation.top_labels[0]
            #Map each explanation weight to the corresponding superpixel
            dict_heatmap = dict(explanation.local_exp[ind])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            # Scale the normalized values to the range [-1, 1]
            heatmap_normalized = heatmap_normalized * 2 - 1        
            # Combine positive and negative masks for a comprehensive view
            combined_mask = mask_p - mask_n  # Positive minus negative attributions


            if num_instances <= 2:
                continue

            for page in range(num_pages):
                # Create the plot for each page
                fig, axes = plt.subplots(images_per_page // images_per_row, images_per_row * num_cols_per_image, figsize=(8.27, 11.69), constrained_layout=True)  # A4 size in inches (landscape)

                for i in range(images_per_page):
                    image_index = page * images_per_page + i
                    if image_index >= num_instances:
                        break
                    
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

                    ax_img = axes[i // images_per_row, (i % images_per_row) * 2]
                    ax_img.imshow(original_image / 255.0, cmap='gray')
                    ax_img.axis('off')
                    ax_img.set_title(f'Image {image_index + 1}')

                    # ax_img_0 = axes[i // images_per_row, (i % images_per_row) * 2 + 1]
                    # ax_img_0.imshow(mask_p[image_index], cmap='gray')
                    # ax_img_0.axis('off')
                    # ax_img_0.set_title(f'P mask')

                    # ax_img_1 = axes[i // images_per_row, (i % images_per_row) * 2 + 2]
                    # ax_img_1.imshow(mask_n[image_index], cmap='gray')
                    # ax_img_1.axis('off')
                    # ax_img_1.set_title(f'N mask')
                    # p_overlay = label2rgb(mask_p[image_index], image=original_image, bg_label=0, alpha=0.4)
                    # p_overlay_with_boundaries = mark_boundaries(p_overlay, mask_p[image_index], mode='inner')
                    # p_final_image = np.where(p_overlay[..., :3] != original_image, p_overlay[..., :3], p_overlay_with_boundaries)
                    p_overlay = original_image.copy()

                    # Apply the color to the regions defined in the mask
                    for region in np.unique(mask_n[image_index]):
                        if region == 0:
                            continue  # Skip the background
                        # Blend the color with the original image
                        if (FLAGS.imgs_type == 'CZ_HE' and image_index < len(cz_img_paths[0])) or FLAGS.imgs_type == 'CZ':
                            highlight_color = np.array([111, 45, 168]) #128, 128, 0])  # Red color in RGB
                        else:
                            highlight_color = np.array([128, 128, 0])

                        p_overlay[mask_n[image_index] == region] = (
                            p_overlay[mask_n[image_index] == region] * 0.6 + highlight_color * 0.4
                        ).astype(np.uint8)

                    # Optionally, add boundaries on top of the overlay
                    p_overlay_with_boundaries = mark_boundaries(p_overlay, mask_p[image_index], color=(1, 1, 1), mode='outer')


                    ax_img_2 = axes[i // images_per_row, (i % images_per_row) * 2 + 1]
                    ax_img_2.imshow(p_overlay_with_boundaries)
                    ax_img_2.axis('off')
                    ax_img_2.set_title(f'Attribution Mask')
                    # Create custom legend for the highlighted and non-highlighted areas

                                        
                    # n_overlay = original_image.copy()
                    # # Apply the color to the regions defined in the mask
                    # for region in np.unique(mask_p[image_index]):
                    #     if region == 0:
                    #         continue  # Skip the background
                    #     # Blend the color with the original image
                    #     n_overlay[mask_p[image_index] == region] = (
                    #         n_overlay[mask_p[image_index] == region] * 0.6 + highlight_color * 0.4
                    #     ).astype(np.uint8)

                    # # Optionally, add boundaries on top of the overlay
                    # n_overlay_with_boundaries = mark_boundaries(n_overlay, mask_n[image_index], color=(1, 1, 1), mode='inner')

                    # # n_overlay = label2rgb(mask_n[image_index], image=original_image, bg_label=0, alpha=0.4)
                    # # n_overlay_with_boundaries = mark_boundaries(n_overlay, mask_n[image_index], mode='inner')
                    # # n_final_image = np.where(n_overlay[..., :3] != original_image, n_overlay[..., :3], n_overlay_with_boundaries)

                    # ax_img_3 = axes[i // images_per_row, (i % images_per_row) * 2 + 2]
                    # ax_img_3.imshow(n_overlay_with_boundaries)
                    # ax_img_3.axis('off')
                    # ax_img_3.set_title(f'N attribution')

                    # Assuming heatmap contains both positive and negative attributions
                    img_heatmap = heatmap_normalized[image_index]
                    # Normalize based on actual range of data
                    vmin = img_heatmap.min()
                    vmax = img_heatmap.max()
                    # Define normalization range

                    # Plot the heatmap with 'RdBu' colormap and apply normalization in imshow
                    ax_img_4 = axes[i // images_per_row, (i % images_per_row) * 2 + 2]
                    cax = ax_img_4.imshow(img_heatmap, cmap='RdBu_r', vmin=vmin, vmax=vmax)  # No need to manually normalize img_heatmap
                    ax_img_4.axis('off')
                    ax_img_4.set_title(f'Segments heatmap')
    
                    cbar = fig.colorbar(cax, ax=ax_img_4, orientation='vertical', location='right', fraction=0.046, pad=0.09,  aspect=30)
                    cbar.set_ticks(np.linspace(vmin, vmax, num=5))
                    # cbar.set_ticklabels([])            

                    cbar.set_ticklabels([f'{x:.1f}' for x in np.linspace(vmin, vmax, num=5)])            

                    # plt.colorbar()
                    # ax_img_3 = axes[i // images_per_row, (i % images_per_row) * 2 + 4]
                    # ax_img_3.imshow(mark_boundaries(original_image / 255.0, mask_n[image_index], mode='subpixel'), cmap='gray') # , color=(1, 0, 0) for red boundaries
                    # ax_img_3.axis('off')
                    # ax_img_3.set_title(f'Negative attribution')  
                    #                   
                    # Plot the bar chart
                    ax_bar = axes[i // images_per_row, (i % images_per_row) * 2 + 3]
                    ax_bar.bar(['GoodProg', 'PoorProg'], [probs_att[0].item(), probs_att[1].item()], color=colors) 

                    ax_bar.set_xlim(0, 1)
                    ax_bar.set_ylim(0, 1)

                    # Remove the right and top borders
                    ax_bar.spines['right'].set_visible(False)
                    ax_bar.spines['top'].set_visible(False)               

                    ax_bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    ax_bar.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Enable y-axis labels
                    ax_bar.set_title(f'0: {probs_att[0].item():.3f}, 1: {probs_att[1].item():.3f}')
                    #Select the same class explained on the figures above.


                for j in range((i // images_per_row) + 1, images_per_page // images_per_row):
                    if j == 5:
                        fig.delaxes(axes[j, 0])
                        fig.delaxes(axes[j, 1])
                        fig.delaxes(axes[j, 2])          
                        fig.delaxes(axes[j, 3])  
                        # fig.delaxes(axes[j, 4])                                   
                    for k in range(images_per_row * num_cols_per_image):
                        fig.delaxes(axes[j-1, k])
        
                # Add a legend to the figure
                if page == 0:
                    old_slide_id = patients_mapping[patients_mapping[:,1] == slide_id, 0]
                    fig.suptitle('Slide: {}, Label: {}, Predicted: {}'.format(old_slide_id[0], lable[0], predicted_arr[0]), fontsize='large', fontweight='bold')

                if page == num_pages - 1:
                    # Create custom legend for probabilities (Good Prog and Poor Prog)
                    legend_patches_probs = [
                        mpatches.Patch(color=colors[0], label='Good Prog {:.4f}'.format(probs_arr[0, 0].item())),
                        mpatches.Patch(color=colors[1], label='Poor Prog {:.4f}'.format(probs_arr[0, 1].item()))
                    ]
                    
                    # Create custom legend for attributions (Positive and Negative)
                    legend_patches_attr = [
                        mpatches.Patch(facecolor='none', edgecolor='black', label='Positive (transparent)'),  # Transparent with black edge
                        mpatches.Patch(color='red', label='Negative (highlighted)')  # Red color for negative attribution
                    ]

                    # Combine both sets of legend patches
                    combined_legend_patches = legend_patches_probs + legend_patches_attr

                    # Adjust the layout of the legend
                    fig.legend(
                        handles=combined_legend_patches, 
                        loc='lower center', 
                        bbox_to_anchor=(0.5, 0.02),  # Adjust the placement
                        ncol=4, 
                        handlelength=2.5, 
                        handleheight=2.5,
                        columnspacing=0.5,  # Reduce the spacing between the columns
                        labelspacing=0.2    # Reduce the space between the text and the boxes
                    )

                pdf_pages.savefig(fig)

            pdf_pages.close()
            print(f"PDF saved with {num_pages} page, num images {num_instances}.\n")

            pbar.update(1)

        pbar.close()


