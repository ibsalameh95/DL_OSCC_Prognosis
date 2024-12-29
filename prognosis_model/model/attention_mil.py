import torch
import torch.nn as nn

from typeguard import typechecked
from typing import Tuple, Optional, Sequence


class StableSoftmax(torch.nn.Module):
    @typechecked
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.nn.LogSoftmax(dim=self.dim)(inputs).exp()

class DefaultAttentionModule(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = nn.ReLU(),
        output_activation: torch.nn.Module = StableSoftmax(dim=1),
        use_batch_norm: bool = False,
        track_bn_stats: bool = False,
        drop_out = nn.Dropout(0.3),
    ):

        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.track_bn_stats = track_bn_stats
        self.drop_out = drop_out

        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [self.output_dims]
        # nodes_by_layer = [self.input_dims] + [self.output_dims]

        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))

        for i, (nodes_in, nodes_out) in iterable:
            layer = nn.Linear(in_features=nodes_in, out_features=nodes_out, bias=True)
            layers.append(layer)
            if i < len(self.hidden_dims):
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(nodes_out, track_running_stats=self.track_bn_stats))
                layers.append(self.hidden_activation)
                layers.append(self.drop_out)
        model = nn.Sequential(*layers)
        print(model)
        input('attention_mil')
        return model

    def forward(self, features, bag_size):
        out = self.model(features)
        out = out.view([-1, bag_size])
        attention = self.output_activation(out)
        return attention.unsqueeze(-1)



class DefaultMILGraph(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        featurizer: torch.nn.Module,
        classifier: torch.nn.Module,
        pointer: torch.nn.Module,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.pointer = pointer

    def forward(self, images: torch.Tensor):
        batch_size, bag_size = 1, images.shape[0]
        shape = [-1] + list(images.shape[1:])  # merge batch and bag dim
        images = images.view(shape)
        features = self.featurizer(images)
        print(features.shape)
        attention = self.pointer(features, bag_size)
        print(attention.shape)
        if not torch.all(attention >= 0):
            raise ValueError("{}: Attention weights cannot be negative".format(attention))

        features = features.view([batch_size, bag_size] + list(features.shape[1:]))  # separate batch and bag dim

        # attention = attention.transpose(0, 2).transpose(1, 2)
        # out_features = torch.mean(features, dim=1)

        classifier_out_dict = self.classifier(features, attention)
        bag_logits = classifier_out_dict['logits']
        print(bag_logits)
        patch_logits = classifier_out_dict['patch_logits'] if 'patch_logits' in classifier_out_dict else None
        print(patch_logits.shape)
        input('sa')
        out = {}
        out['value'] = bag_logits
        if patch_logits is not None:
            out['patch_logits'] = patch_logits
        out['attention'] = attention
        return out
