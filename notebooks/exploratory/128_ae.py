
import torch
from torch import nn
from torch.nn import functional as F
from dii.models import layers
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, stride=1, output=nn.ReLU, pool: int = 0, upsample: int = 0):
        super().__init__()
        layers = []
        if upsample != 0:
            layers.append(nn.Upsample(scale_factor=upsample, mode="bilinear"))
        layers.extend(
            [nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1,),
            nn.BatchNorm2d(num_channels)]
        )
        self.conv = nn.Sequential(*layers)
        if use_1x1conv:
            self.skip = nn.Conv2d(input_channels, num_channels, 1, stride=stride)
        else:
            self.skip = None
        # if activation provided, instantiate the layer
        if output:
            self.output = output()
        if pool != 0:
            self.pool = nn.MaxPool2d(pool)
        else:
            self.pool = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.conv(X)
        if self.skip:
            X = self.skip(X)
        Y.add_(X)
        if self.output:
            Y = self.output(Y)
        if self.pool:
            Y = self.pool(Y)
        return Y


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        sizes = [8, 16, 32, 64, 128,]
        for index, out_channels in enumerate(sizes):
            if index == 0:
                layers = [
                    ResidualBlock(in_channels, out_channels, pool=2, use_1x1conv=True)
                ]
            else:
                layers.append(
                    ResidualBlock(sizes[index - 1], out_channels, pool=2, use_1x1conv=True)
                )
        layers.extend([nn.Flatten(), nn.Linear(128 * 4 * 4, latent_dim)])
        self.model = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        sizes = [128, 64, 32, 16, 8, out_channels]
        self.fc = nn.Linear(latent_dim, sizes[0] * 4 * 4, bias=False)
        model = list()
        for index, out_channels in enumerate(sizes):
            if index == 0:
                pass
            elif index == len(sizes):
                model.append(layers.DecoderBlock(
                    sizes[index - 1], out_channels, 3, upsample_size=2, activation=None
                ))
            else:
                model.append(
                    layers.DecoderBlock(
                        sizes[index - 1], out_channels, 3, upsample_size=2,
                    )
                )
        self.model = nn.Sequential(*model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.fc(X).view(-1, 128, 4, 4)
        return self.model(output)


if __name__ == "__main__":
    model = Encoder(1, 128)
    print(summary(model, (1, 128, 128), device="cpu"))
    model = Decoder(128 *4 * 4, 1)
    print(summary(model, (128 * 4 * 4,), device="cpu"))
    model = nn.Sequential(Encoder(1, 128), Decoder(128, 1))
    print(summary(model, (1, 128, 128), device="cpu"))