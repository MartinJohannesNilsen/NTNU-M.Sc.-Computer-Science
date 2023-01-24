import torch
from torch import nn
from typing import Tuple, List


"""
Possible improvements:
x Lr 
x Adam
x mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
x Anchor boxes of smalles size even smaller?
- Batch norm
- Dropout
- LeakyReLU (replace some of the ReLU with this for faster convergence?)
- Batch size of 16? Have to divide epochs by 2 then, or 64 and 64 epochs?
"""


class _FirstLayer(nn.Sequential):
    def __init__(self, in_channels, middle_channels=[32, 64, 64], out_channels=128, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, stride=stride, padding=padding, out_channels=middle_channels[0], kernel_size=kernel_size),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channels[0], stride=stride, padding=padding, out_channels=middle_channels[1], kernel_size=kernel_size),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channels[1], stride=stride, padding=padding, out_channels=middle_channels[2], kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channels[2], stride=2, padding=padding, out_channels=out_channels, kernel_size=kernel_size),
        )


class _X2_ReLU_Conv2d(nn.Sequential):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=3, stride_one=1, stride_two=2, padding_one=1, padding_two=1):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, stride=stride_one, padding=padding_one, out_channels=middle_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channels, stride=stride_two, padding=padding_two, out_channels=out_channels, kernel_size=kernel_size),
        )


class ImprovedModel(torch.nn.Module):
    def __init__(self,
                 output_channels: List[int],
                 image_channels: int,
                 output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.image_channels = image_channels
        self.output_feature_shape = output_feature_sizes

        self.layer_1 = _FirstLayer(in_channels=image_channels, middle_channels=[32, 64, 64], out_channels=self.out_channels[0])
        self.layer_2 = _X2_ReLU_Conv2d(in_channels=self.out_channels[0], middle_channels=256, out_channels=self.out_channels[1])
        self.layer_3 = _X2_ReLU_Conv2d(in_channels=self.out_channels[1], middle_channels=256, out_channels=self.out_channels[2])
        self.layer_4 = _X2_ReLU_Conv2d(in_channels=self.out_channels[2], middle_channels=128, out_channels=self.out_channels[3])
        self.layer_5 = _X2_ReLU_Conv2d(in_channels=self.out_channels[3], middle_channels=128, out_channels=self.out_channels[4])
        self.layer_6 = _X2_ReLU_Conv2d(in_channels=self.out_channels[4], middle_channels=256, out_channels=self.out_channels[5], padding_two=0)

    def forward(self, x):
        out_features = []
        layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5, self.layer_6]
        out = x
        for layer in layers:
            out = layer(out)
            out_features.append(out)

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            out_channel = self.out_channels[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
