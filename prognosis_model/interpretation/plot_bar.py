import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--slide_list_filename', default='prognosis_model/data/seg12/test.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--metrics_dir', default='Results/prognosis_model/test_metrics/2024_08_16__17_49_30__35/test', help='Text file to write metrics', dest='metrics_dir')

FLAGS = parser.parse_args()

# read slide list
data_arr = np.loadtxt(FLAGS.slide_list_filename, delimiter='\t', comments='#', dtype=str)
slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]
print('num_slides:{}'.format(num_slides))


# Function to read the file and extract the probabilities (Assuming the previous read_data function is used)
def read_data(file_path):
    
    data_arr = np.loadtxt(file_path, delimiter='\t', comments='#', dtype=str)

    # Reshape the array if it's 1-dimensional (i.e., only one line)
    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(1, -1)  # Convert to 2D array with one row

    patch_ids = np.asarray(data_arr[:,0], dtype=int)
    class_0_logits = np.asarray(data_arr[:,1], dtype=float)
    class_1_logits = np.asarray(data_arr[:,2], dtype=float)
    class_0_probs = np.asarray(data_arr[:,3], dtype=float)
    class_1_probs = np.asarray(data_arr[:,4], dtype=float)
                
    return patch_ids, class_0_logits, class_1_logits, class_0_probs, class_1_probs


def plot_prob_bar(slide_id):

    metrics_file = '{}/{}_logits.txt'.format(FLAGS.metrics_dir,slide_id)
    # File path to your data file

    # Read the data
    patch_ids, _, _, class_0_probs, class_1_probs = read_data(metrics_file)

    patch_ids = [patch_id + 1 for patch_id in patch_ids]

    # Plotting the bar chart
    x = np.arange(len(patch_ids))  # X-axis locations for the patches
    width = 0.2  # Adjusted width of the bars for better spacing

    # Determine figure size based on the number of patches
    if len(patch_ids) > 8:
        fig_width = 18 / 2.54  # Convert 18 cm to inches
    else:
        fig_width = 9 / 2.54  # Default width (9 cm) in inches

    fig, ax = plt.subplots(figsize=(fig_width, 9/2.54))  # 9x9 cm figure size

    # Bar plot for class_0_prob (Yellow for late/no relapse)
    rects1 = ax.bar(x - width/1.5, class_0_probs, width, label='Late / No Relapse', color='orange')

    # Bar plot for class_1_prob (Red for early relapse)
    rects2 = ax.bar(x + width/1.5, class_1_probs, width, label='Early Relapse', color='red')

    # Enable the Y-axis (but remove x-axis ticks and labels)
    ax.get_yaxis().set_visible(False)  # Hide the x-axis ticks and labels
    ax.set_ylim(0, 1)  # Set the y-axis limits from 0 to 1

    # Show x-axis ticks and labels for patch_ids
    ax.set_xticks(x)
    ax.set_xticklabels(patch_ids, fontsize=10)

    # Hide all borders (spines) of the chart
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    ax.legend(loc='best', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=9)

    # Add text above bars for class_0_prob (vertical labels, rotated -90 degrees)
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 1.5, height),
                    xytext=(0, 5),  # Adjusted vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontsize=8)  # Vertical alignment

    # Add text above bars for class_1_prob (vertical labels, rotated -90 degrees)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 1.5, height),
                    xytext=(0, 5),  # Adjusted vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontsize=8)  # Vertical alignment



    # Adjust the layout to make space
    plt.tight_layout()

    # Show the plot
    plt.savefig('{}/{}_probs_bar.png'.format(FLAGS.metrics_dir, slide_id), dpi=300)



def plot_logit_bar(slide_id):

    metrics_file = '{}/{}_logits.txt'.format(FLAGS.metrics_dir,slide_id)
    # File path to your data file

    # Read the data
    patch_ids, class_0_logits, class_1_logits, _, _ = read_data(metrics_file)

    patch_ids = [patch_id + 1 for patch_id in patch_ids]

    # Plotting the bar chart
    x = np.arange(len(patch_ids))  # X-axis locations for the patches
    width = 0.2  # Adjusted width of the bars for better spacing

    # Determine figure size based on the number of patches
    if len(patch_ids) > 8:
        fig_width = 18 / 2.54  # Convert 18 cm to inches
    else:
        fig_width = 9 / 2.54  # Default width (9 cm) in inches

    fig, ax = plt.subplots(figsize=(fig_width, 9/2.54))  # 9x9 cm figure size

    # Bar plot for class_0_prob (Yellow for late/no relapse)
    rects1 = ax.bar(x - width/1.5, class_0_logits, width, label='Late / No Relapse', color='orange')

    # Bar plot for class_1_prob (Red for early relapse)
    rects2 = ax.bar(x + width/1.5, class_1_logits, width, label='Early Relapse', color='red')

   # Set y-axis limits dynamically based on the minimum and maximum values
    min_y = min(min(class_0_logits), min(class_1_logits))
    max_y = max(max(class_0_logits), max(class_1_logits))
    ax.get_yaxis().set_visible(False)  # Hide the x-axis ticks and labels
    ax.set_ylim(min_y * 1.1, max_y * 1.1)  # Adding a 10% margin for better visibility

    # Add a horizontal line at y=0 to show the x-axis at 0 level
    # ax.axhline(0, color='black', linewidth=1.3)


    # Set the x-axis to be positioned at y=0
    ax.spines['bottom'].set_position(('data', 0))

    # Ensure the left and right spines are visible
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Optionally, make the x-axis spine more visible
    ax.spines['bottom'].set_linewidth(1)

    ax.set_xticks(x)
    ax.set_xticklabels(patch_ids, fontsize=10)
    # Add gridlines if necessary
    # ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
    # ax.spines['bottom'].set_visible(False)
    # Add gridlines for better separation
    # ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)

    # Add text above/below bars for class_0_prob and class_1_prob (vertical labels, rotated -90 degrees)
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 1.5, height),
                    xytext=(0, 5 if height >= 0 else -10),  # Offset above for positive and below for negative
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', rotation=90, fontsize=8)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 1.5, height),
                    xytext=(0, 5 if height >= 0 else -10),  # Offset above for positive and below for negative
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', rotation=90, fontsize=8)

    # Add the legend at the top center of the plot
    ax.legend(loc='best', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=9)

    # Adjust the layout to make space
    plt.tight_layout()

    # Show the plot
    plt.savefig('{}/{}_logits_bar.png'.format(FLAGS.metrics_dir, slide_id), dpi=300)




def plot_scatter(slide_id):

    metrics_file = '{}/{}_logits.txt'.format(FLAGS.metrics_dir,slide_id)
    # File path to your data file

    # Read the data
    patch_ids, class_0_logits, class_1_logits, _, _ = read_data(metrics_file)

    patch_ids = [patch_id + 1 for patch_id in patch_ids]

    # Plotting the bar chart
    x = np.arange(len(patch_ids))  # X-axis locations for the patches
    width = 0.2  # Adjusted width of the bars for better spacing

    # Determine figure size based on the number of patches
    if len(patch_ids) > 8:
        fig_width = 18 / 2.54  # Convert 18 cm to inches
    else:
        fig_width = 9 / 2.54  # Default width (9 cm) in inches

    # fig, ax = plt.subplots(figsize=(fig_width, 9/2.54))  # 9x9 cm figure size

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust the figure size as needed

    # Plot scatter points
    ax.scatter(class_0_logits, class_1_logits, color='magenta', label='Patch points')

    # Define base starting position for the stacked labels (above and to the right of the points)
    base_x_offset = 10  # Fixed horizontal offset for all labels
    base_y_offset = 10  # Starting position for the first label
    label_spacing = 15  # Vertical spacing between labels

    # Annotate each point in a stacked format with arrowed lines and edges
    for i, txt in enumerate(patch_ids):
        x = class_0_logits[i]
        y = class_1_logits[i]
        
        # Calculate the stacked label position, all in the same direction (top-right)
        label_x = base_x_offset #+ i * label_spacing 
        label_y = base_y_offset - i * label_spacing  # Offset each label by label_spacing
        
        # Draw an arrow from the point to the label with an edge
        ax.annotate(
            txt,
            (x, y),  # Coordinates for the point
            textcoords="offset points",  # Position the text relative to the data point
            xytext=(label_x, label_y),  # Stacked label position
            ha='right', fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=0.5, color='gray', linestyle='--')  # Arrowed line pointing to the point
        )

    # Calculate symmetric limits with a margin to increase the scale
    min_value = min(min(class_0_logits), min(class_1_logits))
    max_value = max(max(class_0_logits), max(class_1_logits))
    symmetric_limit = max(abs(min_value), abs(max_value)) * 1.5 # Increase the scale by 50%

    # Set the same limits for both axes
    ax.set_xlim(-symmetric_limit, symmetric_limit)
    ax.set_ylim(-symmetric_limit, symmetric_limit)

    # Move the x-axis and y-axis to the center
    ax.spines['left'].set_position('zero')   # Move the y-axis spine to x=0
    ax.spines['bottom'].set_position('zero') # Move the x-axis spine to y=0

    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set axis labels
    ax.set_xlabel('late/no relapse', fontsize=9, fontweight='bold', labelpad=10)
    ax.set_ylabel('early relapse', fontsize=9, fontweight='bold', labelpad=10)

    # Adjust the label positions slightly closer to the plot
    ax.xaxis.set_label_coords(0.90, 0.55)  # X-axis label moved to the right just outside the plot
    ax.yaxis.set_label_coords(0.45, 0.35)  # Y-axis label moved to the bottom just outside the plot


    # Adjust the ticks to be on both sides of the axes
    # ax.xaxis.set_ticks_position('both')
    # ax.yaxis.set_ticks_position('both')

    # Add gridlines (optional)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Show plot
    # plt.show()

    plt.savefig('{}/{}_logits_scatter.png'.format(FLAGS.metrics_dir, slide_id), dpi=300)

for s, slide_id in enumerate(slide_ids):
    # print('probs bar for slide {}/{}: {}'.format(s+1,num_slides,slide_id))

    # plot_prob_bar(slide_id)

    # print('logits bar for slide {}/{}: {}'.format(s+1,num_slides,slide_id))

    # plot_logit_bar(slide_id)    

    print('scatter plot for logits for slide {}/{}: {}'.format(s+1,num_slides,slide_id))

    plot_scatter(slide_id)        