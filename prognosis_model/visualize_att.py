import os
import numpy as np
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=256)

def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
	print(title)
	for i in range(len(feature_names)):
		print(feature_names[i], ": ", '%.3f'%(importances[i]))
	x_pos = (np.arange(len(feature_names)))
	if plot:
		f = plt.figure(figsize=(12,6))
		plt.bar(x_pos, importances, align='center')
		plt.xticks(x_pos, feature_names, wrap=True)
		plt.xlabel(axis_title)
		plt.title(title)

		plt.show()

		return f

# read slide list

slide_list_filename = 'prognosis_model/Data/seg2/test_attribution.txt'

data_arr = np.loadtxt(slide_list_filename, delimiter='\t', comments='#', dtype=str)
slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]
model_name = '2024_04_28__14_21_30__352'

for s, slide_id in enumerate(slide_ids):
	print(slide_id)
	data_dir = 'Results/prognosis_model/IG_blurred_images/{}/{}'.format(model_name, slide_id)
	
	if not os.path.exists(data_dir):
			continue
		
	files = [file for file in os.listdir(data_dir) if file.endswith('npy')]
	num_images = int(len(files) / 2)

	for i in range(num_images):

		img_arr = np.load('{}/{}__img.npy'.format(data_dir, i))
		attr_arr = np.load('{}/{}__attribution.npy'.format(data_dir, i)) 
		img_resized = cv2.resize(img_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

		fig, ax = viz.visualize_image_attr_multiple(attr_arr,
												img_resized,
												methods=["heat_map", "original_image"],
												cmap=default_cmap,
												show_colorbar=True,
												signs=["positive", "all"],
												titles=["attribution", "image"],
												fig_size=(8, 6), use_pyplot=False,
												)


		fig.tight_layout()
		fig_filename = '{}/{}_attr_positive_ig.png'.format(data_dir, i)
		fig.savefig(fig_filename, dpi=200)


	for i in range(num_images):

		img_arr = np.load('{}/{}__img.npy'.format(data_dir, i))
		attr_arr = np.load('{}/{}__attribution.npy'.format(data_dir, i)) 
		img_resized = cv2.resize(img_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

		fig, ax = viz.visualize_image_attr_multiple(attr_arr,
												img_resized,
												methods=["heat_map", "original_image"],
												cmap=default_cmap,
												show_colorbar=True,
												signs=["absolute_value", "all"],
												titles=["attribution", "image"],
												fig_size=(8, 6), use_pyplot=False,
												)


		fig.tight_layout()
		fig_filename = '{}/{}_attr_absolute_value_ig.png'.format(data_dir, i)
		fig.savefig(fig_filename, dpi=200)


	for i in range(num_images):

		img_arr = np.load('{}/{}__img.npy'.format(data_dir, i))
		attr_arr = np.load('{}/{}__attribution.npy'.format(data_dir, i)) 
		img_resized = cv2.resize(img_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

		fig, ax = viz.visualize_image_attr_multiple(attr_arr,
												img_resized,
												methods=["blended_heat_map", "original_image"],
												cmap=default_cmap,
												show_colorbar=True,
												signs=["absolute_value", "all"],
												titles=["attribution", "image"],
												fig_size=(8, 6), use_pyplot=False,
												)


		fig.tight_layout()
		fig_filename = '{}/{}_attr_absolute_value_blended_heat_map_ig.png'.format(data_dir, i)
		fig.savefig(fig_filename, dpi=200)


	for i in range(num_images):

		img_arr = np.load('{}/{}__img.npy'.format(data_dir, i))
		attr_arr = np.load('{}/{}__attribution.npy'.format(data_dir, i)) 
		img_resized = cv2.resize(img_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

		fig, ax = viz.visualize_image_attr_multiple(attr_arr,
												img_resized,
												methods=["blended_heat_map", "original_image"],
												cmap=default_cmap,
												show_colorbar=True,
												signs=["positive", "all"],
												titles=["attribution", "image"],
												fig_size=(8, 6), use_pyplot=False,
												)


		fig.tight_layout()
		fig_filename = '{}/{}_attr_positive_blended_heat_map_ig.png'.format(data_dir, i)
		fig.savefig(fig_filename, dpi=200)


            
            
