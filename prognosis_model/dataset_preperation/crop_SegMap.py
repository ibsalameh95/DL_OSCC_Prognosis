import os
import numpy as np
import imageio
import openslide
import matplotlib.pyplot as plt
from PIL import Image
from tiles_operations import getCoordinates, crop_center


HE_slides_filelist = 'prognosis_model/Data/he_filelist'
HE_annotations_filelist = 'prognosis_model/Data/he_annotations'

segmentation_maps_path = 'Data/segmented_slides/StainNormalized_FullDS/Run1/resnet50/from_scratch'

out_path = 'Data/CroppedRegions/SegmentationMapsPNG'

slides = np.loadtxt(HE_slides_filelist, dtype=str, comments='#', delimiter='\t')
annotations = np.loadtxt(HE_annotations_filelist, dtype=str, comments='#', delimiter='\t')

slides_arr = slides.reshape((-1,))
annotations_arr = annotations.reshape((-1,))

num_slides = slides_arr.shape[0]

for i in range(num_slides):

	# image reader
	slide_path = slides_arr[i]
	slide_name = slide_path.split('/')[-1].split('.')[0]

	print('Slide-{}/{}: {}'.format(i+1,num_slides,slide_name))

	idx = [i for i,item in enumerate(annotations_arr) if slide_name in item]
	out_dir = os.path.join(out_path, slide_name)

	annotation_idx = idx[0]
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	slide = openslide.OpenSlide(slide_path)
	read_level = 0 #num_series-2
	factor = 16
	for NAME, X, Y, W, H in getCoordinates(annotations_arr[annotation_idx], 'HE'):
		try:
			# read image at the bottom of the pyramid / highest resolution level
			if read_level == 1: 
				factor = 2
			img_width = int(W / 64)
			img_height = int(H / 64)

			mask_x = int(X / 64)
			mask_y = int(Y / 64)

			img_path = os.path.join(segmentation_maps_path, slide_name, slide_name + '_segmented_wsi_non_blured.png')

			if not os.path.exists(img_path):
				continue

			img = Image.open(img_path).convert("RGB")
			
			img = np.array(img)

			crop = img[mask_y:mask_y + img_height, mask_x: mask_x+ img_width]

			imageio.imwrite('{}/{}.png'.format(out_dir, NAME), crop)

		except Exception as ex:
			print(ex)
			continue
