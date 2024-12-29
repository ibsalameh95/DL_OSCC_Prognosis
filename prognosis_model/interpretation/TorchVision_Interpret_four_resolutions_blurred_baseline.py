# modified from: https://captum.ai/tutorials/Resnet_TorchVision_Interpret
# 03.09.2022 - MUO

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from PIL import ImageFilter

import os
import sys
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from model import Model

# ## 1- Loading the model and the dataset
#

# Loads pretrained model and sets it to eval mode

model = Model(pretrained=False, num_classes=2, num_intermediate_features=64)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

init_model_file = "saved_models/model_weights__2021_06_23__19_47_33__76.pth"
state_dict = torch.load(init_model_file, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict["model_state_dict"])

model = model.eval()

# create outdir if does not exist
model_name = init_model_file.split("/")[-1][15:-4]
# outdir = 'computed_attributions_blurred_baseline_radius15/{}'.format(model_name)
outdir = "computed_attributions_white_baseline/{}".format(model_name)
# outdir = 'computed_attributions_gray_baseline/{}'.format(model_name)
if not os.path.exists(outdir):
    os.makedirs(outdir)

# get the image


# target (t_1): pattern3 - malignant, prediction (p_1): malignant
# img_high_path = '../cropped_patches/patient_057_slide_01_img_8__t_1__p_1__level0.png'
# img_medium_path = '../cropped_patches/patient_057_slide_01_img_8__t_1__p_1__level1.png'
# img_low_path = '../cropped_patches/patient_057_slide_01_img_8__t_1__p_1__level2.png'
# img_low2_path = '../cropped_patches/patient_057_slide_01_img_8__t_1__p_1__level3.png'

# target (t_4): tangential benign - benign, prediction (p_1): malignant
# img_high_path = '../cropped_patches/patient_091_slide_01_img_160__t_4__p_1__level0.png'
# img_medium_path = '../cropped_patches/patient_091_slide_01_img_160__t_4__p_1__level1.png'
# img_low_path = '../cropped_patches/patient_091_slide_01_img_160__t_4__p_1__level2.png'
# img_low2_path = '../cropped_patches/patient_091_slide_01_img_160__t_4__p_1__level3.png'

# target (t_2): pattern4 - malignant, prediction (p_0): benign
# img_high_path = '../cropped_patches/patient_075_slide_01_img_147__t_2__p_0__level0.png'
# img_medium_path = '../cropped_patches/patient_075_slide_01_img_147__t_2__p_0__level1.png'
# img_low_path = '../cropped_patches/patient_075_slide_01_img_147__t_2__p_0__level2.png'
# img_low2_path = '../cropped_patches/patient_075_slide_01_img_147__t_2__p_0__level3.png'

# target (t_4): tangential benign - benign, prediction (p_0): benign
# img_high_path = '../cropped_patches/patient_096_slide_01_img_88__t_4__p_0__level0.png'
# img_medium_path = '../cropped_patches/patient_096_slide_01_img_88__t_4__p_0__level1.png'
# img_low_path = '../cropped_patches/patient_096_slide_01_img_88__t_4__p_0__level2.png'
# img_low2_path = '../cropped_patches/patient_096_slide_01_img_88__t_4__p_0__level3.png'

# target (t_4): tangential benign - benign, prediction (p_0): benign
img_high_path = "../cropped_patches/patient_064_slide_01_img_43__t_4__p_0__level0.png"
img_medium_path = "../cropped_patches/patient_064_slide_01_img_43__t_4__p_0__level1.png"
img_low_path = "../cropped_patches/patient_064_slide_01_img_43__t_4__p_0__level2.png"
img_low2_path = "../cropped_patches/patient_064_slide_01_img_43__t_4__p_0__level3.png"

# target (t_3): benign - benign, prediction (p_0): benign
# not checked by pathologist
# img_high_path = '../cropped_patches/patient_087_slide_01_img_36__t_3__p_0__level0.png'
# img_medium_path = '../cropped_patches/patient_087_slide_01_img_36__t_3__p_0__level1.png'
# img_low_path = '../cropped_patches/patient_087_slide_01_img_36__t_3__p_0__level2.png'
# img_low2_path = '../cropped_patches/patient_087_slide_01_img_36__t_3__p_0__level3.png'

# target (t_1): pattern3 - malignant, prediction (p_1): malignant
# img_high_path = '../cropped_patches/patient_049_slide_01_img_118__level0.png'
# img_medium_path = '../cropped_patches/patient_049_slide_01_img_118__level1.png'
# img_low_path = '../cropped_patches/patient_049_slide_01_img_118__level2.png'
# img_low2_path = '../cropped_patches/patient_049_slide_01_img_118__level3.png'


print("img_high_path: {}".format(img_high_path))
print("img_medium_path: {}".format(img_medium_path))
print("img_low_path: {}".format(img_low_path))
print("img_low2_path: {}".format(img_low2_path))


img_high = Image.open(img_high_path).convert("RGB")
img_medium = Image.open(img_medium_path).convert("RGB")
img_low = Image.open(img_low_path).convert("RGB")
img_low2 = Image.open(img_low2_path).convert("RGB")

transformed_img_high = TF.center_crop(img_high, 362)
transformed_img_medium = TF.center_crop(img_medium, 362)
transformed_img_low = TF.center_crop(img_low, 362)
transformed_img_low2 = TF.center_crop(img_low2, 362)

transformed_img_high = TF.to_tensor(transformed_img_high)
transformed_img_medium = TF.to_tensor(transformed_img_medium)
transformed_img_low = TF.to_tensor(transformed_img_low)
transformed_img_low2 = TF.to_tensor(transformed_img_low2)

input_high = transformed_img_high.unsqueeze(0)
input_medium = transformed_img_medium.unsqueeze(0)
input_low = transformed_img_low.unsqueeze(0)
input_low2 = transformed_img_low2.unsqueeze(0)

input_high = input_high.to(device)
input_medium = input_medium.to(device)
input_low = input_low.to(device)
input_low2 = input_low2.to(device)


baseline_high = img_high.filter(ImageFilter.BoxBlur(radius=15))
baseline_medium = img_medium.filter(ImageFilter.BoxBlur(radius=15))
baseline_low = img_low.filter(ImageFilter.BoxBlur(radius=15))
baseline_low2 = img_low2.filter(ImageFilter.BoxBlur(radius=15))

transformed_baseline_high = TF.center_crop(baseline_high, 362)
transformed_baseline_medium = TF.center_crop(baseline_medium, 362)
transformed_baseline_low = TF.center_crop(baseline_low, 362)
transformed_baseline_low2 = TF.center_crop(baseline_low2, 362)

transformed_baseline_high = TF.to_tensor(transformed_baseline_high)
transformed_baseline_medium = TF.to_tensor(transformed_baseline_medium)
transformed_baseline_low = TF.to_tensor(transformed_baseline_low)
transformed_baseline_low2 = TF.to_tensor(transformed_baseline_low2)

basel_high = transformed_baseline_high.unsqueeze(0)
basel_medium = transformed_baseline_medium.unsqueeze(0)
basel_low = transformed_baseline_low.unsqueeze(0)
basel_low2 = transformed_baseline_low2.unsqueeze(0)

# white baseline
# basel_high = torch.ones(1,3,362,362)
# basel_medium = torch.ones(1,3,362,362)
# basel_low = torch.ones(1,3,362,362)
# basel_low2 = torch.ones(1,3,362,362)

# gray baseline
# basel_high = torch.ones(1,3,362,362)*0.5
# basel_medium = torch.ones(1,3,362,362)*0.5
# basel_low = torch.ones(1,3,362,362)*0.5
# basel_low2 = torch.ones(1,3,362,362)*0.5

# green baseline
# basel_high = torch.zeros(1,3,362,362)
# basel_high[0,1,:,:] = 0.
# basel_medium = torch.zeros(1,3,362,362)
# basel_medium[0,1,:,:] = 0.
# basel_low = torch.zeros(1,3,362,362)
# basel_low[0,1,:,:] = 0.
# basel_low2 = torch.zeros(1,3,362,362)
# basel_low2[0,1,:,:] = 0.

# random uniform baseline
# basel_high = torch.rand(1,3,362,362)
# basel_medium = torch.rand(1,3,362,362)
# basel_low = torch.rand(1,3,362,362)
# basel_low2 = torch.rand(1,3,362,362)

# added Gaussian noise
# basel_high = torch.clamp(basel_high + torch.normal(mean=0.,std=3.,size=(1,3,362,362)),min=0.,max=1.)
# basel_medium = torch.clamp(basel_medium + torch.normal(mean=0.,std=3.,size=(1,3,362,362)),min=0.,max=1.)
# basel_low = torch.clamp(basel_low + torch.normal(mean=0.,std=3.,size=(1,3,362,362)),min=0.,max=1.)
# basel_low2 = torch.clamp(basel_low2 + torch.normal(mean=0.,std=3.,size=(1,3,362,362)),min=0.,max=1.)

basel_high = basel_high.to(device)
basel_medium = basel_medium.to(device)
basel_low = basel_low.to(device)
basel_low2 = basel_low2.to(device)

# Predict the class of the input image
output = model(input_high, input_medium, input_low, input_low2)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

print(
    "Input - Predicted: {} ({:.4f})".format(
        pred_label_idx.squeeze().item(), prediction_score.squeeze().item()
    )
)

# Predict the class of the baseline image
output = model(basel_high, basel_medium, basel_low, basel_low2)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

print(
    "Baseline - Predicted: {} ({:.4f})".format(
        pred_label_idx.squeeze().item(), prediction_score.squeeze().item()
    )
)

# sys.exit()

# ## 2- Gradient-based attribution

# Let's compute attributions using Integrated Gradients and visualize them on the image.
# Integrated gradients computes the integral of the gradients of the output of the model
# for the predicted class `pred_label_idx` with respect to the input image pixels along
# the path from the black image to our input image.

integrated_gradients = IntegratedGradients(model)
(
    attributions_ig_high,
    attributions_ig_medium,
    attributions_ig_low,
    attributions_ig_low2,
) = integrated_gradients.attribute(
    (input_high, input_medium, input_low, input_low2),
    (basel_high, basel_medium, basel_low, basel_low2),
    target=pred_label_idx,
    n_steps=200,
    internal_batch_size=20,
)
# attributions_ig_high,attributions_ig_medium,attributions_ig_low,attributions_ig_low2 = integrated_gradients.attribute((input_high,input_medium,input_low,input_low2), target=0, n_steps=200, internal_batch_size=20)


# save attributions
attributions_high_path = "{}/{}.npy".format(outdir, img_high_path.split("/")[-1][:-4])
attributions_medium_path = "{}/{}.npy".format(
    outdir, img_medium_path.split("/")[-1][:-4]
)
attributions_low_path = "{}/{}.npy".format(outdir, img_low_path.split("/")[-1][:-4])
attributions_low2_path = "{}/{}.npy".format(outdir, img_low2_path.split("/")[-1][:-4])

np.save(
    attributions_high_path,
    np.transpose(attributions_ig_high.squeeze().cpu().detach().numpy(), (1, 2, 0)),
)
np.save(
    attributions_medium_path,
    np.transpose(attributions_ig_medium.squeeze().cpu().detach().numpy(), (1, 2, 0)),
)
np.save(
    attributions_low_path,
    np.transpose(attributions_ig_low.squeeze().cpu().detach().numpy(), (1, 2, 0)),
)
np.save(
    attributions_low2_path,
    np.transpose(attributions_ig_low2.squeeze().cpu().detach().numpy(), (1, 2, 0)),
)
