import torch.nn as nn
import torch
import torchvision


####################
###    Task 2    ###
####################

class Task2Model(nn.Module):
    def __init__(self, image_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # Layer 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )
        self.num_output_features = 128*4*4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        out = self.classifier(features)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


####################
###    Task 3    ###
####################

class ModelS(nn.Module):
    def __init__(self, image_channels, num_classes, conv_kernel_size=5):
        super().__init__()

        # Model Parameters
        self.conv_kernel = conv_kernel_size
        self.conv_padding = 2
        if conv_kernel_size == 3:
            self.conv_padding = 1
        self.conv_stride = 1
        self.pooling_kernel = 2
        self.pooling_stride = 2
        self.hidden_layer_units = 64
        self.num_classes = num_classes

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=self.pooling_kernel,
                stride=self.pooling_stride
            ),
            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=self.pooling_kernel,
                stride=self.pooling_stride
            ),
            # Layer 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=self.pooling_kernel,
                stride=self.pooling_stride
            ),
            # Layer 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )
        self.num_output_features = 256 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, self.hidden_layer_units),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_units, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        out = self.classifier(features)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class ModelE(nn.Module):
    def __init__(self, image_channels, num_classes, conv_kernel_size=5):
        super().__init__()

        # Model Parameters
        self.conv_kernel = conv_kernel_size
        self.conv_padding = 2
        self.conv_stride = 1
        self.pooling_kernel = 2
        self.pooling_stride = 2
        self.hidden_layer_units = 64
        self.num_classes = num_classes

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(
                kernel_size=self.pooling_kernel,
                stride=self.pooling_stride
            ),
            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(
                kernel_size=self.pooling_kernel,
                stride=self.pooling_stride
            ),
            # Layer 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(
                kernel_size=self.pooling_kernel,
                stride=self.pooling_stride
            ),
            # Layer 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )
        self.num_output_features = 256 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, self.hidden_layer_units),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(self.hidden_layer_units, num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        out = self.classifier(features)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        # No need to apply softmax, as this is done in nn.CrossEntropyLoss
        self.model.fc = nn.Linear(512, 10)
        for param in self.model.parameters():  # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters():  # Unfreeze the last fully-connected layer
            param.requires_grad = True
        for param in self.model.layer4.parameters():  # Unfreeze the last 5 convolutional layers
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x
