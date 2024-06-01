import os
import numpy as np
import imageio
import openslide
import matplotlib.pyplot as plt
from PIL import Image
from tiles_operations import getCoordinates, cropCenter

HE_slides_filelist = 'prognosis_model/Data/he_filelist1'
HE_annotations_filelist = 'prognosis_model/Data/he_annotations'

out_path = 'Data/CroppedRegions/WholeImage/BrightfieldTest'

slides = np.loadtxt(HE_slides_filelist, dtype=str, comments='#', delimiter='\t')
annotations = np.loadtxt(HE_annotations_filelist, dtype=str, comments='#', delimiter='\t')

slides_arr = slides.reshape((-1,))
annotations_arr = annotations.reshape((-1,))

num_slides = slides_arr.shape[0]

crop_width = 1024
crop_height = 1024

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
	read_level = 1 #num_series-2

	for NAME, X, Y, W, H in getCoordinates(annotations_arr[annotation_idx], 'HE'):

		try:
			# read image at the bottom of the pyramid / highest resolution level

			if read_level == 2:

				W = int(W/2)
				H = int(H/2)

			img_size = (W, H)

			im = slide.read_region((X, Y), read_level, img_size)

			img = np.array(im)[:, :, 0:3]

			h,w, _ = img.shape 

			img_rgb = np.zeros((h,w,3), dtype=np.uint8)
			img_rgb[:,:,:3] = img 
			
			img_cropped = img_rgb #cropCenter(img_rgb, crop_width, crop_height)

			imageio.imwrite('{}/{}.png'.format(out_dir, NAME), img_cropped)

		except Exception as ex:
			print('Img size exceeded')
			continue