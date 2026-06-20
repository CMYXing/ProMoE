import torch
import torch.nn as nn


def kaiming_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ConvBnRelu3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, activate: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activate = activate
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return self.relu(x) if self.activate else x


class ResidualBlock3d(nn.Module):
    def __init__(self, channels: int, num_convs: int, bottleneck: bool = False, ratio: int = 4):
        super().__init__()
        layers = []
        for idx in range(num_convs):
            activate = idx != num_convs - 1
            if bottleneck:
                hidden = max(channels // ratio, 1)
                layers.extend(
                    [
                        ConvBnRelu3d(channels, hidden, 1, 0, True),
                        ConvBnRelu3d(hidden, hidden, 3, 1, True),
                        ConvBnRelu3d(hidden, channels, 1, 0, activate),
                    ]
                )
            else:
                layers.append(ConvBnRelu3d(channels, channels, 3, 1, activate))
        self.ops = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.ops(x))


class InputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBnRelu3d(in_channels, out_channels, 3, 1, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, num_convs: int, bottleneck: bool = False):
        super().__init__()
        out_channels = in_channels * 2
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock3d(out_channels, num_convs=num_convs, bottleneck=bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int, bottleneck: bool = False):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock3d(out_channels, num_convs=num_convs, bottleneck=bottleneck)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = _center_or_pad_to(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.residual(x)


class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _center_or_pad_to(x: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    target_d, target_h, target_w = shape
    d, h, w = x.shape[2:]
    pad_d = max(target_d - d, 0)
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_d or pad_h or pad_w:
        x = nn.functional.pad(
            x,
            [
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_d // 2,
                pad_d - pad_d // 2,
            ],
        )
    d, h, w = x.shape[2:]
    start_d = max((d - target_d) // 2, 0)
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    return x[:, :, start_d : start_d + target_d, start_h : start_h + target_h, start_w : start_w + target_w]

