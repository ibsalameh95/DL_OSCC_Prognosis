import argparse
from datetime import datetime
import os
import math
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
# from model import Model
from model.additive_model import get_additive_mil_model
from tqdm import tqdm
from dataset.dataset import Dataset, custom_collate_fn, worker_init_fn
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--init_model_file', default='Results/prognosis_model/saved_models/model_weights__2024_08_16__17_49_30__35.pth',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--slide_list_filename', default='prognosis_model/data/seg/test.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--patch_size', default='512', type=int, help='Patch size', dest='patch_size')
parser.add_argument('--num_instances', default='0', type=int, help='number of instances (patches) in a bag', dest='num_instances')
parser.add_argument('--num_bags', default='100', type=int, help='number of instances (patches) in a bag', dest='num_bags')
parser.add_argument('--num_features', default='64', type=int, help='number of features', dest='num_features')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='1', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--metrics_dir', default='Results/prognosis_model/test_metrics/', help='Text file to write metrics', dest='metrics_dir')
parser.add_argument('--cz_imgs_list', default='prognosis_model/data/cz_img_list.txt', help='Image directory', dest='cz_imgs_list')
parser.add_argument('--he_imgs_list', default='prognosis_model/data/he_img_list.txt', help='Image directory', dest='he_imgs_list')
parser.add_argument('--imgs_type', default='CZ_HE', help='Image directory', dest='imgs_type')

FLAGS = parser.parse_args()




def create_image_grid(image_paths, padding=2):
    """
    Creates a grid of images from a list of image paths, dynamically calculating grid size,
    resizes images to 128x128, returns a single concatenated image with black space between the images,
    and plots the result.
    
    Args:
    - image_paths: List of paths to the images.
    - padding: Space (in pixels) between images (default is 10px).
    
    Returns:
    - A single image that contains the grid of all the input images with padding.
    """
    num_images = len(image_paths)
    
    # Dynamically calculate grid size (round up to the nearest integer)
    grid_size = math.ceil(math.sqrt(num_images))
    
    # Resize all images to 128x128
    img_width, img_height = 128, 128
    images = [Image.open(img_path).resize((img_width, img_height)) for img_path in image_paths]
    
    # Create a new blank image for the grid (grid_size * grid_size), accounting for padding
    grid_img_width = img_width * grid_size + padding * (grid_size - 1)
    grid_img_height = img_height * grid_size + padding * (grid_size - 1)
    grid_img = Image.new('RGB', (grid_img_width, grid_img_height), color='black')  # Black background
    
    # Arrange images in the grid with padding
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        grid_img.paste(img, (x, y))
        
    return grid_img


def create_prediction_map(patch_logits, patch_size=128, padding=2):
    """
    Creates a prediction map from the patch logits with padding between patches.
    
    Args:
    - patch_logits: List of tuples, where each tuple contains (logit_class_0, logit_class_1) for each patch.
    - patch_size: The size of each patch in the final map (e.g., 5 pixels).
    - padding: The number of pixels for black space between patches (default is 2 pixels).
    
    Returns:
    - A 3D numpy array representing the prediction map with padding.
    """
    
    num_patches = len(patch_logits)
    
    # Dynamically calculate grid size (round up to the nearest integer)
    grid_size = int(np.ceil(np.sqrt(num_patches)))
    
    # Calculate the size of the prediction map, accounting for padding
    map_size = grid_size * (patch_size + padding) - padding  # Total size with padding in between
    
    # Initialize the prediction map with zeros (black background)
    prediction_map = np.zeros((map_size, map_size, 3))
        
    # Fill the prediction map with logits
    for i, (logit_class_0, logit_class_1) in enumerate(patch_logits):
        row = i // grid_size
        col = i % grid_size
        
        # Calculate the top-left corner for each patch, accounting for padding
        row_start = row * (patch_size + padding)
        col_start = col * (patch_size + padding)
        
        # Set the second channel with class 0 logits
        prediction_map[row_start:row_start+patch_size, col_start:col_start+patch_size, 1] = logit_class_0
        
        # Set the third channel with class 1 logits
        prediction_map[row_start:row_start+patch_size, col_start:col_start+patch_size, 2] = logit_class_1

    return prediction_map

def prediction_map_to_image(prediction_map):
    """
    Converts a prediction map (NumPy array) into a PIL image.
    
    Args:
    - prediction_map: A 3D numpy array representing the prediction map.
    
    Returns:
    - A PIL Image.
    """
    # Normalize the prediction map to 0-255 for visualization
    prediction_map_normalized = (prediction_map * 255 / np.max(prediction_map)).astype(np.uint8)

    # Convert to PIL Image
    prediction_map = Image.fromarray(prediction_map_normalized)

    return prediction_map


def apply_cmap_to_prediction_map(prediction_map):

    if not isinstance(prediction_map, np.ndarray):
        prediction_map = np.array(prediction_map)
    # Get the min and max logits for both class 0 and class 1
    min_logit_class_0 = np.min(prediction_map[:, :, 1])
    max_logit_class_0 = np.max(prediction_map[:, :, 1])
    min_logit_class_1 = np.min(prediction_map[:, :, 2])
    max_logit_class_1 = np.max(prediction_map[:, :, 2])

    # Normalize logits between their min and max for class 0 (blue)
    norm_class_0 = (prediction_map[:, :, 1] - min_logit_class_0) / (max_logit_class_0 - min_logit_class_0)

    # Normalize logits between their min and max for class 1 (green)
    norm_class_1 = (prediction_map[:, :, 2] - min_logit_class_1) / (max_logit_class_1 - min_logit_class_1)

    # Create custom colormaps for each class
    cmap_green = LinearSegmentedColormap.from_list('GreenScale', ['black', 'green'])
    cmap_blue = LinearSegmentedColormap.from_list('BlueScale', ['black', 'blue'])

    # Apply the colormap for each class
    colored_map_class_0 = cmap_green(norm_class_0)
    colored_map_class_1 = cmap_blue(norm_class_1)

    # Combine the two maps
    combined_map = colored_map_class_0 + colored_map_class_1

    # Plot the combined map
    plt.imshow(combined_map)
    
    # Generate a scalar mappable based on the colors of the combined map
    combined_map_flat = combined_map.reshape(-1, 4)  # Flatten the color map to apply to the colorbar

    combined_map_flat = np.clip(combined_map_flat, 0, 1)
    
    unique_colors = np.unique(combined_map_flat, axis=0)  # Get unique colors from the map
    
    # Create a colormap based on the unique colors in the prediction map
    color_map_custom = LinearSegmentedColormap.from_list('CustomMap', unique_colors)
    
    return color_map_custom



# def plot_prediction_with_images(prediction_map, grid_img, patch_logits, image_paths, patch_size, old_slide_id, slide_label, predicted, out_path):

#     """
#     Plots the prediction map with a color scale from green to blue.
    
#     Args:
#     - prediction_map: A 3D numpy array representing the prediction map.
#     - min_logit: The minimum logit value across all patches.
#     - max_logit: The maximum logit value across all patches.
#     """
#     # Convert prediction map to image
#     prediction_image = prediction_map_to_image(prediction_map)

#     # Find the minimum and maximum logits for each class
#     if slide_label == 0:

#         vmin = min(logit[0] for logit in patch_logits)
#         vmax = max(logit[0] for logit in patch_logits)

#         # cmap = LinearSegmentedColormap.from_list("BlackGreen", ["black", "green"])
#         cmap = LinearSegmentedColormap.from_list("BlueGreen", ["blue", "green"])

#         # cmap = apply_cmap_to_prediction_map(prediction_map=prediction_image)
        

#     if slide_label == 1:

#         vmin = min(logit[1] for logit in patch_logits)
#         vmax = max(logit[1] for logit in patch_logits) 

#         # cmap = LinearSegmentedColormap.from_list("BlackBlue", ["black", "blue"])
#         cmap = LinearSegmentedColormap.from_list("GreenBlue", ["green", "blue"])
#         # cmap = apply_cmap_to_prediction_map(prediction_map=prediction_image)

#     # prediction_map = add_border_to_prediction_map(prediction_map, 5, 1)
#     num_rows = prediction_map.shape[0] // patch_size
#     num_cols = prediction_map.shape[1] // patch_size
    
#     # Create a grid for both the prediction map and the matrix of images
#     fig = plt.figure(figsize=(10, 10))
#     gs = gridspec.GridSpec(num_rows, num_cols, width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)

#     # Plot the images in the grid
#     for i, image_path in enumerate(image_paths):
#         img = Image.open(image_path)
#         row = i // num_cols
#         col = i % num_cols
#         ax = plt.subplot(gs[row, col])
#         ax.imshow(img)
#         ax.axis('off')  # Hide axis for images
#     plt.subplots_adjust(wspace=0, hspace=0)  # No horizontal or vertical space between images
#     # Add the prediction map on the side
#     fig.add_subplot(gs[:, :])
#     plt.imshow(prediction_map, cmap=cmap, vmin=vmin, vmax=vmax)
#     plt.axis('off')

#     # Add the colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
#     sm.set_array([])
#     cbar = plt.colorbar(sm, orientation='vertical', fraction=0.046, pad=0.04)
    
#     # Set colorbar labels
#     if slide_label == 0:
#         cbar.ax.text(1.5, 1.03, 'Good Prognosis', ha='center', va='center', transform=cbar.ax.transAxes, fontsize=10)
#     if slide_label == 1:
#         cbar.ax.text(1.5, 1.03, 'Poor Prognosis', ha='center', va='center', transform=cbar.ax.transAxes, fontsize=10)

#     # Set plot title
#     plt.title('Slide: {}, Label: {}, Predicted: {}'.format(old_slide_id[0], slide_label, predicted[0]), fontsize='large', fontweight='bold')

#     plt.axis('off')  # Turn off the axis
#     plt.tight_layout()
#     plt.show()


def plot_prediction_map_with_legend(prediction_map, grid_img, patch_logits, probs, old_slide_id, slide_label, predicted, out_path):
    """
    Plots the prediction map with a color scale from green to blue.
    
    Args:
    - prediction_map: A 3D numpy array representing the prediction map.
    - min_logit: The minimum logit value across all patches.
    - max_logit: The maximum logit value across all patches.
    """
    # Convert prediction map to image
    prediction_image = prediction_map_to_image(prediction_map)

    good_prog = min(logit[0] for logit in probs)
    poor_prog = min(logit[1] for logit in probs)

    # Find the minimum and maximum logits for each class
    if slide_label == 0:

        min_logit = min(logit[0] for logit in patch_logits)
        max_logit = max(logit[0] for logit in patch_logits)

        # cmap = LinearSegmentedColormap.from_list("BlackGreen", ["black", "green"])
        cmap = LinearSegmentedColormap.from_list("BlueGreen", ["blue", "green"])

        # cmap = apply_cmap_to_prediction_map(prediction_map=prediction_image)
        

    if slide_label == 1:

        min_logit = min(logit[1] for logit in patch_logits)
        max_logit = max(logit[1] for logit in patch_logits) 

        # cmap = LinearSegmentedColormap.from_list("BlackBlue", ["black", "blue"])
        cmap = LinearSegmentedColormap.from_list("GreenBlue", ["green", "blue"])
        # cmap = apply_cmap_to_prediction_map(prediction_map=prediction_image)
    # Display the image using Matplotlib


    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].imshow(grid_img)
    ax[0].axis('off')
    # ax[0].set_title('Slide Patches', fontsize='large', fontweight='bold')

    ax[1].imshow(prediction_image)
    ax[1].axis('off')
    # ax[1].set_title('Slide: {}, Label: {}, Predicted: {}'.format(old_slide_id[0], slide_label, predicted[0]), fontsize='large', fontweight='bold')
    # plt.imshow(prediction_image)
    
    # Create a colorbar below the images
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_logit, vmax=max_logit))
    sm.set_array([])
    
    # Place the colorbar horizontally below both subplots
    cbar = plt.colorbar(sm, orientation='horizontal', fraction=0.046, pad=0.09, ax=ax[1], aspect=30)

    cmap_hidden = LinearSegmentedColormap.from_list("GreenBlue", ["green", "blue"])
    sm_hidden = plt.cm.ScalarMappable(cmap=cmap_hidden, norm=plt.Normalize(vmin=good_prog, vmax=poor_prog))
    sm_hidden.set_array([])
    cbar_hidden = plt.colorbar(sm_hidden, orientation='horizontal', fraction=0.046, pad=0.09, ax=ax[0], aspect=30)
    cbar_hidden.ax.set_visible(False)

    # cbar_hidden.ax.text(-1.1, 1.4, '{:.4f}'.format(good_prog.item()), ha='left', va='center', transform=cbar.ax.transAxes, fontsize=10)
    # cbar_hidden.ax.text(-0.2, 1.4, '{:.4f}'.format(poor_prog.item()), ha='left', va='center', transform=cbar.ax.transAxes, fontsize=10)

    # cbar_hidden.ax.text(0.81, 1.4, poor_prog, ha='left', va='center', transform=cbar.ax.transAxes, fontsize=10)
    # Optional: Set custom tick labels or text
    if slide_label == 0:
        cbar.ax.text(0.71, 1.4, 'Good Prognosis: {:.4f}'.format(good_prog.item()), ha='left', va='center', transform=cbar.ax.transAxes, fontsize=10)
    elif slide_label == 1:
        cbar.ax.text(0.71, 1.4, 'Poor Prognosis: {:.4f}'.format(poor_prog.item()), ha='left', va='center', transform=cbar.ax.transAxes, fontsize=10)

    # cbar.set_label("Logit Value", rotation=270, labelpad=15)
    # plt.title('Slide: {}, Label: {}, Predicted: {}'.format(old_slide_id[0], slide_label, predicted[0]), fontsize='large', fontweight='bold')

    plt.suptitle('Slide: {}, Label: {}, Predicted: {}'.format(old_slide_id[0], slide_label, predicted[0]), fontsize='large', fontweight='bold')
    
    plt.tight_layout(pad=0.5)
    # plt.show()
    plt.savefig(out_path)
    
model_name = FLAGS.init_model_file.split('/')[-1][15:-4]

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

# get the model using helper function
# model = Model(num_classes=FLAGS.num_classes, features_size=1024, num_features=FLAGS.num_features, requires_grad= False)
model = get_additive_mil_model(num_classes=FLAGS.num_classes, features_size=1024, num_features=FLAGS.num_features, requires_grad= False)

# move model to the right device
model.to(device)

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
slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]
print('num_slides:{}'.format(num_slides))

patients_mapping = np.loadtxt('Data/patients_mapping.txt', delimiter='\t', comments='#', dtype=str) 


out_dir = '{}/{}/{}'.format(FLAGS.metrics_dir,model_name,FLAGS.slide_list_filename.split('/')[-1][:-4])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model.eval()
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

        old_slide_id = patients_mapping[patients_mapping[:,1] == slide_id, 0]


        for img, lable, cz_img_paths, he_img_paths in data_loader:

            img = img.to(device)
            # get logits from the model
            output = model(img)
            
            # print(output['value'])
            patch_logits = output['patch_logits'] #.squeeze(0).cpu()

            patch_logits_probs = F.softmax(patch_logits, dim=2).squeeze(0).cpu()

            output = output['value']
            # obtain probs
            probs = F.softmax(output, dim=1)
            
            _, predicted = torch.max(output, 1)
            # Create the prediction map
            prediction_map = create_prediction_map(patch_logits_probs, patch_size=128)

            # Convert the prediction map to a PIL image
            prediction_image = prediction_map_to_image(prediction_map)

            # Find the minimum and maximum logits for legend
            min_logit = min(min(logit[0] for logit in patch_logits_probs), min(logit[1] for logit in patch_logits_probs))
            max_logit = max(max(logit[0] for logit in patch_logits_probs), max(logit[1] for logit in patch_logits_probs))

            # Plot the prediction map with legends
            if FLAGS.imgs_type == 'CZ':
                images_grid = create_image_grid(cz_img_paths[0])
            if FLAGS.imgs_type == 'HE':
                images_grid = create_image_grid(he_img_paths[0])

            if FLAGS.imgs_type == 'CZ_HE':                             
                image_paths = np.concatenate((cz_img_paths[0], he_img_paths[0])) 
                images_grid = create_image_grid(image_paths)

            plot_prediction_map_with_legend(prediction_map, images_grid, patch_logits_probs, probs, old_slide_id, slide_label, predicted, out_path= '{}/{}_prediction_map.png'.format(out_dir, slide_id))

            metrics_file = '{}/{}_logits.txt'.format(out_dir, slide_id)
            with open(metrics_file, 'w') as f:
                f.write('# slide_id: {}\n'.format(slide_id))
                f.write('# old_slide_id: {}\n'.format(old_slide_id[0]))
                f.write('# good prog: {:.5f}\n'.format(probs[0, 0].item()))
                f.write('# good logit: {:.5f}\n'.format(output[0, 0].item()))
                f.write('# poor prog: {:.5f}\n'.format(probs[0, 1].item()))
                f.write('# poor logit: {:.5f}\n'.format(output[0, 1].item()))                
                f.write('# patch_id\tclass_0_logit\tclass_1_logit\tclass_0_prob\tclass_1_prob\n')

            patch_logits = patch_logits.cpu().detach().numpy()
            patch_logits_probs = patch_logits_probs.cpu().detach().numpy()
            
            with open(metrics_file, 'a') as f:
                # Loop over the tensors and write each pair of values from both tensors on the same line
                for idx, (row1, row2) in enumerate(zip(patch_logits[0], patch_logits_probs)):  # Loop over both tensors
                    #f.write(f'{row1[0]:.4f}\t{row1[1]:.4f}\t{row2[0]:.4f}\t{row2[1]:.4f}\n')
                    f.write(f'{idx}\t{row1[0]:.5f}\t{row1[1]:.5f}\t{row2[0]:.5f}\t{row2[1]:.5f}\n')                

            pbar.update(1)

        pbar.close()
        



