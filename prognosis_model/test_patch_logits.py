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
from model.additive_model import get_additive_mil_model
from tqdm import tqdm
from dataset.dataset import Dataset, custom_collate_fn, worker_init_fn


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--init_model_file', default='Results/prognosis_model/saved_models/model_weights__2024_08_19__17_31_46__369.pth',help='Initial model file (optional)', dest='init_model_file')
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
    
model_name = FLAGS.init_model_file.split('/')[-1][15:-4]

out_dir = '{}/{}/{}'.format(FLAGS.metrics_dir,model_name,FLAGS.slide_list_filename.split('/')[-1][:-4])
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


model.eval()
with torch.no_grad():

    for s, slide_id in enumerate(slide_ids):
        print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

        slide_label = labels[s]

        # dataset for the current slide
        dataset = Dataset(slide_list_filename= FLAGS.slide_list_filename, slide_id=slide_id, cz_imgs_list= FLAGS.cz_imgs_list, he_imgs_list= FLAGS.he_imgs_list, imgs_type= FLAGS.imgs_type)

        # define data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)

        len(data_loader)
        
        metrics_file = '{}/test_scores__{}.txt'.format(out_dir,slide_id)
        with open(metrics_file, 'w') as f:
            f.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
            f.write('# model_name: {}\n'.format(model_name))
            f.write('# slide_list_filename: {}\n'.format(FLAGS.slide_list_filename))
            f.write('# patch_size: {}\n'.format(FLAGS.patch_size))
            f.write('# imgs_type: {}\n'.format(FLAGS.imgs_type))            
            f.write('# num_instances: {}\n'.format(FLAGS.num_instances))
            f.write('# num_features: {}\n'.format(FLAGS.num_features))
            f.write('# num_bags: {}\n'.format(FLAGS.num_bags))
            f.write('# num_classes: {}\n'.format(FLAGS.num_classes))
            f.write('# batch_size: {}\n'.format(FLAGS.batch_size))
            f.write('# slide_id\tbag_id\tinstance_id\tinstance_path\tscore_negative\tscore_positive\tnegative_logit\tpositive_logit\n')

        # if dataset.num_patches == 0:
        #     for idx in range(FLAGS.num_bags):
        #         with open(metrics_file, 'a') as f:
        #             f.write('{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\n'.format(slide_id, idx, slide_label, 0, 1.0, 0.0))

        #     continue

        torch.set_printoptions(profile="full")
        bag_count = 0
        pbar = tqdm(total=len(data_loader))
        for img, lable, cz_img_paths, he_img_paths in data_loader:

            img = img.to(device)
            # get logits from the model
            output = model(img)
            
            # print(output['value'])
            patch_logits = output['patch_logits'].squeeze(0)
            
            patch_probs = F.softmax(output['patch_logits'], dim=2).squeeze(0)

            # sum_probs_neg = patch_probs[0, :, 0].sum()
            # sum_probs_pos = patch_probs[0, :, 1].sum()

            # input('ds')
            output = output['value']
            # obtain probs
            probs = F.softmax(output, dim=1)

            # scaled_neg_probs = patch_probs[0, :, 0] * (probs[0,0] / sum_probs_neg)
            # scaled_pos_probs = patch_probs[0, :, 1] * (probs[0,1] / sum_probs_pos)
            # scaled_patch_probs = torch.stack([scaled_neg_probs, scaled_pos_probs], dim=1)

            # obtain predictions
            _, predicted = torch.max(output, 1)

            predicted_arr = predicted.cpu().numpy()
            probs_arr = probs.cpu().numpy()

            num_instances = img.shape[0]

            temp_num_predictions = predicted_arr.shape[0]


            for idx in range(num_instances):
                if FLAGS.imgs_type == 'CZ_HE':
                    if idx < len(cz_img_paths[0]):
                        instance_path = cz_img_paths[0][idx]

                    if idx >= len(cz_img_paths[0]):
                        instance_path = he_img_paths[0][idx-num_instances]

                if FLAGS.imgs_type == 'CZ':
                    instance_path = cz_img_paths[0][idx]

                if FLAGS.imgs_type == 'HE':
                    instance_path = he_img_paths[0][idx]

                with open(metrics_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(slide_id, bag_count, idx + 1, instance_path, patch_probs[idx,0], patch_probs[idx,1], patch_logits[idx,0], patch_logits[idx,1])) #probs_arr[idx, 0], probs_arr[idx, 1]))

            with open(metrics_file, 'a') as f:
                f.write('# slide_id\tbag_id\tslide_label\tprediction\tscore_negative\tscore_positive\tnegative_logit\tpositive_logit\n')
            for idx in range(temp_num_predictions):
                with open(metrics_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(slide_id, bag_count, slide_label, predicted_arr[idx], probs_arr[idx, 0], probs_arr[idx, 1], output[idx,0], output[idx,1]))

                # bag_count += 1 

            pbar.update(1)

        pbar.close()