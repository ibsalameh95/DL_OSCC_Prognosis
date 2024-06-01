import os
import json
import numpy as np 
from PIL import Image

def get_xywh(coordinates, downsample):
    x_values = [downsample * point[0] for point in coordinates]
    y_values = [downsample * point[1] for point in coordinates]

    x = min(x_values)
    y = min(y_values)
    width = max(x_values) - x
    height = max(y_values) - y

    return x, y, width, height

# Replace 'your_file.json' with the path to your JSON file


def getCoordinates(file_path, slide_type):
    coordinates_list = []

    if slide_type == 'czi':
        # downsample = 0.5
        downsample = 1
    else:
        downsample = 1

    with open(file_path, 'r') as file:
        data = json.load(file)

    features = data['features']

    for feature in features:
        geometry = feature['geometry']
        properties = feature['properties']
        coordinates = geometry['coordinates'][0]  # Extracting the coordinates from the 'geometry' key
        x, y, w, h = get_xywh(coordinates, downsample)
        name = properties['name']
        
        coordinates_list.append((name, x, y, int(w), int(h)))
        # print(f"Name: {name}, X: {x}, Y: {y}, Width: {w}, Height: {h}")

    return coordinates_list


def cropCenter(img_rgb, crop_width, crop_height):

        img_arr = np.array(img_rgb)

        img_height, img_width = img_arr.shape[:2]
        
        if img_width >= crop_width and img_height >= crop_height:

            start_x = max(img_width//2 - (crop_width//2), 0)
            start_y = max(img_height//2 - (crop_height//2), 0)
            end_x = min(start_x + crop_width, img_width)
            end_y = min(start_y + crop_height, img_height)
            
            cropped_img = img_arr[start_y:end_y, start_x:end_x]
            
            return cropped_img




def get_counts(wsi_filelist, root_dir):
    # Load all slide ids
    wsi_data = np.loadtxt(wsi_filelist, delimiter='\t', comments='#', dtype=str)
    good_prog_slides = wsi_data[wsi_data[:, 1] == '0'][:,0].tolist()
    poor_prog_slides = wsi_data[wsi_data[:, 1] == '1'][:,0].tolist()


    num_files = 0
    for slide in good_prog_slides:
        dir = os.path.join(root_dir, slide)
        num_files += len(os.listdir(dir))

    print('GoodPrognosis: {}'.format(num_files))

    num_files = 0
    for slide in poor_prog_slides:
        dir = os.path.join(root_dir, slide)
        num_files += len(os.listdir(dir))

    print('PoorPrognosis: {}'.format(num_files))        


def resizeImages():
    for root, dnames, fnames in os.walk('Data/CroppedRegions/WholeImage/Confocal'):
        for fname in fnames:
            if fname.endswith('png'):

                path = os.path.join(root, fname)

                out_dir = root.replace('Confocal', 'ConfocalResized')

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                # Open the image
                img = Image.open(path)

                current_resolution_CF = 0.3900
                # current_resolution_BF = 0.2204

                desired_resotution = 0.50

                scaling_factor_CF = current_resolution_CF / desired_resotution
                # scaling_factor_BF = current_resolution_BF / desired_resotution

                # Resize the image
                new_width = int(img.width * scaling_factor_CF)
                new_height = int(img.height * scaling_factor_CF)
                resized_img = img.resize((new_width, new_height))

                # Save the resized image
                out_file = os.path.join(out_dir, fname)

                resized_img.save(out_file)



def getSmallImages():
    cnt = 0

    for root, dnames, fnames in os.walk('Data/CroppedRegions/WholeImage/ConfocalResized'):
        for fname in fnames:
            if fname.endswith('png'):

                path = os.path.join(root, fname)

                img = Image.open(path)

                if img.width < 512 or img.height < 512:
                    cnt +=1
                    print(path)
    print(cnt)
                    


# getSmallImages()                    
# get_counts('prognosis_model/Data/all.txt', 'Data/CroppedRegions/512/Confocal')
