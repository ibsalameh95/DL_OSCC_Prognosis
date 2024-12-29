import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append('./prognosis_model/model')

from attention_mil import DefaultAttentionModule, DefaultMILGraph
from additive_mil import AdditiveClassifier

torch.set_printoptions(precision=6, threshold=20, edgeitems=3)
np.set_printoptions(formatter={'float': lambda x: "{:5.6f}".format(x)})

class FeatureVectors(nn.Module):
    def __init__(self, features_size = 1024, num_features= 64, requires_grads= False):
        super(FeatureVectors, self).__init__()
        self._features_size = features_size

        original_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT', progress=True)

        children_list = list(original_model.children())[:-3]
        children_list.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.feature_extractor = nn.Sequential(*children_list)

        for param in self.feature_extractor.parameters():
            param.requires_grad = requires_grads

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(FeatureVectors, self).train(mode)

        # print("Freezing Mean/Var of BatchNorm2D.")
        # print("Freezing Weight/Bias of BatchNorm2D.")

        for m in self.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
 
        features_out = self.feature_extractor(x)

        features_out = features_out.reshape(-1, self._features_size)
    

        return features_out


def get_additive_mil_model(features_size= 1024, num_classes=10, num_features=32, device_num= 0, requires_grad= False):
    
    model = DefaultMILGraph(
        featurizer = FeatureVectors(features_size= features_size, num_features=num_features, requires_grads= requires_grad),
        pointer= DefaultAttentionModule(hidden_activation = nn.ReLU(), hidden_dims=[num_features], input_dims=1024, output_dims=1),
        classifier= AdditiveClassifier(hidden_dims=[num_features], input_dims=1024, output_dims=2)
    )
    return model