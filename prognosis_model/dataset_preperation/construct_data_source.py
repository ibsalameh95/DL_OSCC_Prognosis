import numpy as np    
import os 

with open('prognosis_model/data/cz_imgs_list.txt', 'w') as all:
    all.write('# slide_id\timg_id\tlabel\tpath\n')

base_dir = 'Data/CroppedRegions/WholeImageCrops/Confocal'

data_arr = np.loadtxt('prognosis_model/data/all.txt', delimiter='\t', comments='#', dtype=str)

slide_ids = data_arr[:,0]
labels = data_arr[:,1]

for idx, slide in enumerate(slide_ids):
    imgs_dir = os.path.join(base_dir, slide)

    imgs = os.listdir(imgs_dir)

    for i, img_path in enumerate(imgs):
        with open('prognosis_model/data/cz_imgs_list.txt', 'a') as all:
            all.write(slide + '\t' + str(i) + '\t' + str(labels[idx]) + '\t' + os.path.join(imgs_dir, img_path) + '\n')