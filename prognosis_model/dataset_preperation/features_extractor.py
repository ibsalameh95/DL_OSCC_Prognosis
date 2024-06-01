import argparse
import os
import sys
sys.path.append('./')
import numpy as np

import torch
import torchvision
import torch.utils.data
from dataset_slide import Dataset
from time import time
from tools.load_model import get_model


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--crops_info', default='Data/CroppedRegions/Test/cropped_patches_info.txt', help='count of cropped regions for each slide', dest='crops_info')
parser.add_argument('--image_dir', default='Data/CroppedRegions/Test/512/Confocal', help='Image directory', dest='image_dir')
parser.add_argument('--slides', default='Data/CroppedRegions/Test/tst.txt', help='slides ', dest='slides')
parser.add_argument('--batch_size', default='1', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--features_dir', default='Data/CroppedRegions/Test/_Extracted_Features', help='Text file to write metrics', dest='features_dir')

FLAGS = parser.parse_args()

out_dir = FLAGS.features_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


if __name__ == '__main__':

    model = get_model(2, 'resnet50_fe', '')
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # trained_model_path = 'Results/StainNormalized_FullDS/Run1/trained_models/resnet50_from_scratch_1.pth'

    # state_dict = torch.load(trained_model_path, map_location=device)

    # # Remove the last classification layer weights
    # del state_dict['fc.weight']
    # del state_dict['fc.bias']

    # model.load_state_dict(state_dict, strict=False)

    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    print('cuda is available: {} \n'.format(torch.cuda.is_available()))
    print('moving model to available device')

    model.to(device)
    model.eval()

    start = time()
    
    data_arr = np.loadtxt(FLAGS.slides, delimiter='\t', comments='#', dtype=str)

    slide_ids = data_arr[:,0]
    slide_labels = np.asarray(data_arr[:,1], dtype=int)


    num_slides = slide_ids.shape[0]
    print('num_slides:{}'.format(num_slides))

    with torch.no_grad():

        for s, slide_id in enumerate(slide_ids):
            print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

            # dataset for the current slide
            dataset = Dataset(image_dir=FLAGS.image_dir, img_ext = 'png', crops_info= FLAGS.crops_info, slide_id=slide_id, transforms=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms())

            # define data loader
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0)


            features_file = '{}/features__{}.txt'.format(out_dir,slide_id)

            if dataset.num_patches == 0:
                continue

            try:
                for i,img in enumerate(data_loader):
                    img = img.to(device)

                    # get logits from the model
                    features_out = model(img).cpu().numpy()
                    print(features_out)
                    input('ds')
                    with open(features_file, 'ab') as f_features_file:
                        np.savetxt(f_features_file, features_out.reshape((-1,1024)), fmt='%5.6f', delimiter='\t')
            except Exception as ex:
                print(ex)
                continue

    # start training and evaluation

    train_time = ((time()-start)/60/60)

    print(train_time)