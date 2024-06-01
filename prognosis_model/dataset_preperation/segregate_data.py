import pandas as pd
import numpy as np
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_filelist', type=str, default='segmentation_model/Data/all.txt', help='The name of file containing the WSI Ids.')
FLAGS = parser.parse_args()

random.seed(42)

# Load previously segregated data (to keep the same segregation from the previous model)

previous_train_file = np.loadtxt('segmentation_model/Data/seg3/train.txt', delimiter='\t', comments='#', dtype=str)
previous_train_file = previous_train_file[previous_train_file[:, 1] != '1'] # exclude non-tumor slides
previous_valid_file = np.loadtxt('segmentation_model/Data/seg3/valid.txt', delimiter='\t', comments='#', dtype=str)
previous_valid_file = previous_valid_file[previous_valid_file[:, 1] != '1'] # exclude non-tumor slides
previous_test_file = np.loadtxt('segmentation_model/Data/seg3/test.txt', delimiter='\t', comments='#', dtype=str)
previous_test_file = previous_test_file[previous_test_file[:, 1] != '1'] # exclude non-tumor slides

previous_train_ids = previous_train_file[:,0]
previous_train_labels = np.asarray(previous_train_file[:,1], dtype=int)

previous_valid_ids = previous_valid_file[:,0]
previous_valid_labels = np.asarray(previous_valid_file[:,1], dtype=int)

previous_test_ids = previous_test_file[:,0]
previous_test_labels = np.asarray(previous_test_file[:,1], dtype=int)

# Load all slide ids
wsi_data = np.loadtxt(FLAGS.wsi_filelist, delimiter='\t', comments='#', dtype=str)
wsi_data = wsi_data[wsi_data[:, 1] != '1']
num_wsi = len(wsi_data)

# Remove slide ids that were previously segregated
remaining_slide_ids = list(set(wsi_data[:,0]) - set(previous_train_ids) - set(previous_valid_ids) - set(previous_test_ids))

# Filter the first column based on string values
filtered_indices = np.isin(wsi_data[:,0], remaining_slide_ids)
# Filter the entire array based on filtered indices
filtered_wsi_data = wsi_data[filtered_indices]


wsi_ids = filtered_wsi_data[:,0]
wsi_labels = np.asarray(filtered_wsi_data[:,1], dtype=int)

# Define proportions for training, validation, and testing sets
train_ratio = 0.7
valid_ratio = 0.1
test_ratio = 1.0 - (train_ratio + valid_ratio)

# Shuffle the remaining slide ids
negative_class_indices = np.asarray(wsi_labels == 0).nonzero()[0] # good prognosis
num_negatives = len(negative_class_indices)

positive_class_indices = np.asarray(wsi_labels == 2).nonzero()[0] # poor prognosis
num_positives = len(positive_class_indices)

random.shuffle(negative_class_indices)
random.shuffle(positive_class_indices)


# Split the remaining slide ids into train, valid, and test sets

num_negatives_train = int(num_negatives*train_ratio)
num_negatives_valid = int(num_negatives*valid_ratio)
num_negatives_test = num_negatives - (num_negatives_train + num_negatives_valid)

num_positives_train = int(num_positives*train_ratio)
num_positives_valid = int(num_positives*valid_ratio)
num_positives_test = num_positives - (num_positives_train + num_positives_valid)


# Write the new train, valid, and test files
output_dir = 'prognosis_model/Data/seg3'

os.makedirs(output_dir, exist_ok=True)

#train
with open(os.path.join(output_dir, 'train.txt'), 'w') as f_train_dataset_file:
    f_train_dataset_file.write('# slide_id\tlabel\n')

    for i,ind in enumerate(previous_train_ids):

        slide_id = previous_train_ids[i]
        label = previous_train_labels[i]

        if label == 0: # Good Prgo --> Good Prgo
            f_train_dataset_file.write('{}\t{}\n'.format(slide_id,0))

        if label == 2: # Poor Prog --> Poor Prog
            f_train_dataset_file.write('{}\t{}\n'.format(slide_id,1))

    for i,ind in enumerate(negative_class_indices[:num_negatives_train]):
        slide_id = wsi_ids[ind]
        label = 0
        f_train_dataset_file.write('{}\t{}\n'.format(slide_id,label))

    for i,ind in enumerate(positive_class_indices[:num_positives_train]):
        slide_id = wsi_ids[ind]
        label = 1
        f_train_dataset_file.write('{}\t{}\n'.format(slide_id,label))
            

# valid
with open(os.path.join(output_dir, 'valid.txt'), 'w') as f_valid_dataset_file:
    f_valid_dataset_file.write('# slide_id\tlabel\n')
    
    for i,ind in enumerate(previous_valid_ids):

        slide_id = previous_valid_ids[i]
        label = previous_valid_labels[i]

        if label == 0: # Good Poor Prgo --> Tumor
            f_valid_dataset_file.write('{}\t{}\n'.format(slide_id,0))

        if label == 2: # NonTumor --> NonTumor
            f_valid_dataset_file.write('{}\t{}\n'.format(slide_id,1))

    for i,ind in enumerate(negative_class_indices[num_negatives_train:num_negatives_train+num_negatives_valid]):
        slide_id = wsi_ids[ind]
        label = 0
        f_valid_dataset_file.write('{}\t{}\n'.format(slide_id,label))

    for i,ind in enumerate(positive_class_indices[num_positives_train:num_positives_train+num_positives_valid]):
        slide_id = wsi_ids[ind]
        label = 1
        f_valid_dataset_file.write('{}\t{}\n'.format(slide_id,label)) 

# test
with open(os.path.join(output_dir, 'test.txt'), 'w') as f_test_dataset_file:
    f_test_dataset_file.write('# slide_id\tlabel\n')
    
    for i,ind in enumerate(previous_test_ids):

        slide_id = previous_test_ids[i]
        label = int(previous_test_labels[i])
      
        if label == 0: # Good Poor Prgo --> Tumor
            f_test_dataset_file.write('{}\t{}\n'.format(slide_id,0))

        if label == 2: # NonTumor --> NonTumor
            f_test_dataset_file.write('{}\t{}\n'.format(slide_id,1))

    for i,ind in enumerate(negative_class_indices[num_negatives_train+num_negatives_valid:]):
        slide_id = wsi_ids[ind]
        label = 0
        f_test_dataset_file.write('{}\t{}\n'.format(slide_id,label))

    for i,ind in enumerate(positive_class_indices[num_positives_train+num_positives_valid:]):
        slide_id = wsi_ids[ind]
        label = 1
        f_test_dataset_file.write('{}\t{}\n'.format(slide_id,label))
        
