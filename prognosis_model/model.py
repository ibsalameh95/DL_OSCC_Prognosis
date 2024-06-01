import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_printoptions(precision=6, threshold=20, edgeitems=3)
np.set_printoptions(formatter={'float': lambda x: "{:5.6f}".format(x)})

class FeatureVectors(nn.Module):
    def __init__(self, features_size = 1024, num_features=64, requires_grads= False):
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
        
        num_imgs = x.shape[0]
        slide_features = torch.empty(0, self._features_size).cuda(x.get_device())
        
        for idx in range(num_imgs):
            features_out = self.feature_extractor(x[idx].unsqueeze(0))
            features_out = features_out.reshape(-1, self._features_size)

            slide_features = torch.cat((slide_features, features_out), dim = 0)

        return slide_features
            
class FeatureExtractor(nn.Module):
    def __init__(self, features_size = 1024, num_features=64):
        super(FeatureExtractor, self).__init__()

        self.feature_nn = nn.Sequential(
            nn.Linear(features_size, num_features),
            nn.ReLU(),
            )
        
    def forward(self, x):

        features_out = self.feature_nn(x)
        
        return features_out

class SlideClassifier(nn.Module):
    def __init__(self, num_features=32, num_classes=10):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_features, num_classes)
            )

    def forward(self, x):

        out = self.fc(x)

        return out


class Model(nn.Module):

    def __init__(self, features_size= 1024, num_classes=10, num_features=32, device_num= 0, requires_grad= False):
        super(Model, self).__init__()
        self._num_classes = num_classes
        self._num_features = num_features
        self._features_size = features_size
        self._device_num = device_num

        self._feature_vectors = FeatureVectors(features_size= features_size, num_features=num_features, requires_grads= requires_grad)

        self._feature_extractor = FeatureExtractor(features_size= features_size, num_features=num_features)

        self._slide_classifier = SlideClassifier(num_features=num_features, num_classes=num_classes)

    def forward(self, x):

        feature_vectors = self._feature_vectors(x)

        extracted_features = self._feature_extractor(feature_vectors)

        extracted_features = torch.reshape(extracted_features,(1, -1, self._num_features))

        out = torch.mean(extracted_features, dim=1)

        out = self._slide_classifier(out)

        return out