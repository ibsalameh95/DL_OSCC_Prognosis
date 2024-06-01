import os
import numpy as np
import imageio
import javabridge
import bioformats
import matplotlib.pyplot as plt
from PIL import Image
from tiles_operations import getCoordinates, cropCenter

import cv2 

def init_logger(self):
     pass

bioformats.init_logger = init_logger
javabridge.start_vm(class_path=bioformats.JARS,run_headless=True)

czi_slides_filelist = 'prognosis_model/Data/czi_filelist2'
czi_annotations_filelist = 'prognosis_model/Data/czi_annotations'

out_path = 'Data/CroppedRegions/WholeImage/Confocal'

slides = np.loadtxt(czi_slides_filelist, dtype=str, comments='#', delimiter='\t')
annotations = np.loadtxt(czi_annotations_filelist, dtype=str, comments='#', delimiter='\t')

slides_arr = slides.reshape((-1,))
annotations_arr = annotations.reshape((-1,))

num_slides = slides_arr.shape[0]
print(num_slides)
crop_width = 1024
crop_height = 1024

for i in range(num_slides):
	# image reader
	slide_path = slides_arr[i]
	slide_name = slide_path.split('/')[-1].split('.')[0]

	already_cropped = os.listdir('Data/CroppedRegions/WholeImage/Confocal')
	
	print('Slide-{}/{}: {}'.format(i+1,num_slides,slide_name))

	if slide_name in already_cropped:
		print('Skipping {}'.format(slide_name))
		continue

	idx = [i for i,item in enumerate(annotations_arr) if slide_name in item]
	out_dir = os.path.join(out_path, slide_name)

	annotation_idx = idx[0]
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	
	reader = bioformats.get_image_reader('my_image_reader',slide_path)
	reader.rdr.setSeries(0)
	# # num levels in the pyramid
	num_series = reader.rdr.getSeriesCount()

	# get number of channels
	num_channels = reader.rdr.getSizeC()

	current_level = reader.rdr.getSeries()

	read_level = current_level #num_series-2

	print('current_level:{}'.format(current_level))

	
	for NAME, X, Y, W, H in getCoordinates(annotations_arr[annotation_idx], 'czi'):
		try:
			
			# if read_level == 1:
			# 	X = X/2
			# 	Y = Y/2 
			# 	W = int(W/2)
			# 	H = int(H/2)

			img,max_intensity = reader.read(c=None, z=0, t=0, series=read_level, index=None, rescale=False, wants_max_intensity=True, channel_names=None, XYWH=(X, Y, W, H))
			
			if num_channels == 1:
				h,w = img.shape

				backtorgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
				img_rgb = np.zeros((h,w,3), dtype=np.uint8)
				img_rgb[:,:,:3] = backtorgb

					
			else: 
				h,w, _ = img.shape
				img_rgb = np.zeros((h,w,3), dtype=np.uint8)
				img_rgb[:,:,:2] = img 

			img_cropped = img_rgb #cropCenter(img_rgb, crop_width, crop_height)

			imageio.imwrite('{}/{}.png'.format(out_dir, NAME), img_cropped)

		except Exception as ex:
			print(ex)
			input('Wait..')
			continue


javabridge.kill_vm()



