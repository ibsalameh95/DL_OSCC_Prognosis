import argparse
import os
import sys
sys.path.append('./')
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from time import time
import pandas as pd
from tqdm import tqdm, trange
from tools.load_model import get_model


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--slide_list_filename', default='prognosis_model/Data/all.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--batch_size', default='64', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--out_dir', default='Data/CroppedRegions/512/ExtractedFeatures/HE_SEG', help='Text file to write metrics', dest='features_dir')

FLAGS = parser.parse_args()

out_dir = FLAGS.features_dir

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


if __name__ == '__main__':

    czi_fearutes_path = 'Data/CroppedRegions/512/Confocal/_Extracted_Features'
    he_features_path = 'Data/CroppedRegions/512/Brightfield/_Extracted_Features'
    seg_features_path = 'Data/CroppedRegions/512/BrightfieldSegmented/_Extracted_Features'
    
    
    data_arr = np.loadtxt(FLAGS.slide_list_filename, delimiter='\t', comments='#', dtype=str)
    slide_ids = data_arr[:,0]
    labels = np.asarray(data_arr[:,1], dtype = int)

   
    num_slides = slide_ids.shape[0]
    print('num_slides:{}'.format(num_slides))


    for s, slide_id in enumerate(slide_ids):
        try:
            print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

            czi_features_file = '{}/features__{}.txt'.format(czi_fearutes_path,slide_id)
            he_features_file = '{}/features__{}.txt'.format(he_features_path,slide_id)
            seg_features_file = '{}/features__{}.txt'.format(seg_features_path,slide_id)


            he_features = np.loadtxt(he_features_file, delimiter='\t', comments='#', dtype=float)
            seg_features = np.loadtxt(seg_features_file, delimiter='\t', comments='#', dtype=float)



            concatinated = np.concatenate((he_features, seg_features))

            # if os.path.exists(czi_features_file):
            #     czi_fearutes = np.loadtxt(czi_features_file, delimiter='\t', comments='#', dtype=float)

            #     concatinated = np.concatenate((seg_features, czi_fearutes))

            features_file = '{}/features__{}.txt'.format(out_dir,slide_id)


            with open(features_file, 'ab') as f_features_file:
                np.savetxt(f_features_file, concatinated.reshape((-1,1024)), fmt='%5.6f', delimiter='\t')
        except Exception as ex:
            print(ex)
            continue
