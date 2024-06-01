import argparse
from datetime import datetime
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from model import Model
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--slide_list_filename', default='prognosis_model/Data/seg2/test.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--metrics_dir', default='Results/prognosis_model/test_metrics/2024_04_28__14_21_37__107_HE_2/test/slide_scores.txt', help='Text file to write metrics', dest='metrics_dir')

FLAGS = parser.parse_args()

# read slide list
data_arr = np.loadtxt(FLAGS.slide_list_filename, delimiter='\t', comments='#', dtype=str)
slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]
print('num_slides:{}'.format(num_slides))

out_file = '{}/slide_scores.txt'.format(FLAGS.metrics_dir)
with open(out_file, 'w') as f_out_file:
    f_out_file.write('# slide_id\tslide_label\tslide_score_pos\n')


    for s, slide_id in enumerate(slide_ids):
        print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

        slide_label = labels[s]

        # read metrics file 
        # slide_id\tbag_id\tslide_label\tprediction\tscore_negative\tscore_positive\n
        metrics_file = '{}/test_scores__{}.txt'.format(FLAGS.metrics_dir,slide_id)
        
        data = np.loadtxt(metrics_file, delimiter='\t', comments='#', dtype=str).reshape((1,-1))
        score_pos_arr = np.asarray(data[:,-1], dtype=float)

        slide_score_pos = np.amax(score_pos_arr)

        f_out_file.write('{}\t{}\t{}\n'.format(slide_id,slide_label,slide_score_pos))





        