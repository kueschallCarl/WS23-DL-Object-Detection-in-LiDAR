import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

"""Medium Model"""
config_medium = [
    (16, 3, 1),  # Reduced filters
    (32, 3, 2),
    ["B", 1],  # Less repeats
    (64, 3, 2),
    ["B", 1],  # Less repeats
    (128, 3, 2),
    ["B", 2],  # Significantly reduced the complexity here
    (256, 3, 2),
    ["B", 2],  # And here
    (512, 3, 2),
    ["B", 1],  # Reduced repeats for complexity
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
    (64, 1, 1),
    "U",
    (64, 1, 1),
    (128, 3, 1),
    "S",
]

class CNNBlock_medium(nn.Module):
    """
    Convolutional block with optional batch normalization and LeakyReLU.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn_act (bool): Whether to apply batch normalization and LeakyReLU.

    Methods:
        forward(x): Forward pass of the CNN block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn_act (bool): Whether to apply batch normalization and LeakyReLU.
        **kwargs: Additional keyword arguments for convolution layers.
    """
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock_medium(nn.Module):
    """
    Residual block with optional skip connection.

    Attributes:
        channels (int): Number of input and output channels.
        use_residual (bool): Whether to use the residual connection.
        num_repeats (int): Number of times to repeat the internal convolution blocks.

    Methods:
        forward(x): Forward pass of the residual block.

    Args:
        channels (int): Number of input and output channels.
        use_residual (bool): Whether to use the residual connection.
        num_repeats (int): Number of times to repeat the internal convolution blocks.
    """
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock_medium(channels, channels // 2, kernel_size=1),
                    CNNBlock_medium(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction_medium(nn.Module):
    """
    Scale prediction block for YOLO.

    Attributes:
        in_channels (int): Number of input channels.
        num_classes (int): Number of object classes.

    Methods:
        forward(x): Forward pass of the scale prediction block.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of object classes.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock_medium(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock_medium(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3_medium(nn.Module):
    """
    YOLOv3 architecture.

    Attributes:
        in_channels (int): Number of input channels.
        num_classes (int): Number of object classes.

    Methods:
        forward(x): Forward pass of the YOLOv3 model.
        _create_conv_layers(): Create convolutional layers based on the configuration.

    Args:
        in_channels (int): Number of input channels (default is 3 for RGB images).
        num_classes (int): Number of object classes (default is 80 for COCO dataset).
    """
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction_medium):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock_medium) and layer.num_repeats in [2, 8]:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                if len(route_connections) == 0:
                    raise ValueError("Trying to concatenate with an empty route_connections list.")
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config_medium:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock_medium(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock_medium(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock_medium(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock_medium(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction_medium(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers
    
    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)






"""Large Model"""


config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    """
    CNN block consisting of convolution, batch normalization, and LeakyReLU.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn_act (bool): Whether to apply batch normalization and LeakyReLU.

    Methods:
        forward(x): Forward pass of the CNN block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn_act (bool): Whether to apply batch normalization and LeakyReLU.
        **kwargs: Additional keyword arguments for convolution layers.
    """
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Residual block with optional skip connection.

    Attributes:
        channels (int): Number of input and output channels.
        use_residual (bool): Whether to use the residual connection.
        num_repeats (int): Number of times to repeat the internal convolution blocks.

    Methods:
        forward(x): Forward pass of the residual block.

    Args:
        channels (int): Number of input and output channels.
        use_residual (bool): Whether to use the residual connection.
        num_repeats (int): Number of times to repeat the internal convolution blocks.
    """
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    """
    Scale prediction block for YOLO.

    Attributes:
        in_channels (int): Number of input channels.
        num_classes (int): Number of object classes.

    Methods:
        forward(x): Forward pass of the scale prediction block.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of object classes.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    """
    YOLOv3 architecture.

    Attributes:
        in_channels (int): Number of input channels.
        num_classes (int): Number of object classes.

    Methods:
        forward(x): Forward pass of the YOLOv3 model.

    Args:
        in_channels (int): Number of input channels (default: 3).
        num_classes (int): Number of object classes (default: 80).
    """
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers


"""
Run this script for a foward pass test
"""
if __name__ == "__main__":
    num_classes = 1
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
