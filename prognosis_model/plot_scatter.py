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

parser.add_argument('--slide_list_filename', default='prognosis_model/data/seg/test.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--metrics_dir', default='Results/prognosis_model/test_metrics/2024_08_16__18_03_11__66/test', help='Text file to write metrics', dest='metrics_dir')

FLAGS = parser.parse_args()

# read slide list
data_arr = np.loadtxt(FLAGS.slide_list_filename, delimiter='\t', comments='#', dtype=str)
slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]
print('num_slides:{}'.format(num_slides))


patients_mapping = np.loadtxt('Data/patients_mapping.txt', delimiter='\t', comments='#', dtype=str) 

for s, slide_id in enumerate(slide_ids):
    print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

    slide_label = labels[s]

    # read metrics file 
    # slide_id\tbag_id\tslide_label\tprediction\tscore_negative\tscore_positive\n
    metrics_file = '{}/test_scores__{}.txt'.format(FLAGS.metrics_dir,slide_id)
    
    data = np.loadtxt(metrics_file, delimiter='\t', comments='#', dtype=str)

    slide_scores = data[-1].reshape(-1,1)

    instance_scores = data[:-1]

    patch_negative_logits = instance_scores[:, 6].astype(float)
    patch_positive_logit = instance_scores[:, 7].astype(float)

    slide_negative_logits = slide_scores[6].astype(float)
    slide_positive_logits = slide_scores[7].astype(float)

    predicted = slide_scores[3].astype(int)
    # data = np.loadtxt(FLAGS.metrics_file, delimiter='\t', comments='#', dtype=str)
    old_slide_id = patients_mapping[patients_mapping[:,1] == slide_id, 0]

    # negative_logits = data[:, 3].astype(float)
    # positive_logits = data[:, 4].astype(float)

    # Plot the scatter chart
    plt.figure(figsize=(10, 8))

    # Scatter plot, color-coded by label if available
    plt.scatter(patch_negative_logits, patch_positive_logit, color='blue', label='Patches', alpha=0.7)
    plt.scatter(slide_negative_logits, slide_positive_logits, color='red', label='Slides', s=100, edgecolor='black')

    # Set labels and title
    plt.xlabel('Good Prognosis')
    plt.ylabel('Poor Prognosis')

    plt.title('Slide: {}, Label: {}, Predicted: {}'.format(old_slide_id[0], slide_label, predicted[0]), fontsize='large', fontweight='bold')
    plt.legend()
    plt.grid(True)
    # Add a legend if color-coding by label

    # Show plot
    plt.savefig('{}/{}.png'.format(FLAGS.metrics_dir, slide_id))
