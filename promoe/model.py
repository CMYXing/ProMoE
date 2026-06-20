from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DownBlock, InputBlock, OutputBlock, UpBlock, kaiming_init
from .pe2 import PrototypeRouter


class CTConditionedFiLM(nn.Module):
    """AaFM: use global CT context to FiLM-modulate PET features."""

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.beta = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def reset_to_identity(self) -> None:
        # Start from an exact identity transform so CT conditioning is learned gradually.
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, pet_feat: torch.Tensor, ct_feat: torch.Tensor) -> torch.Tensor:
        ct_context = F.adaptive_avg_pool3d(ct_feat, 1)
        gamma = self.gamma(ct_context)
        beta = self.beta(ct_context)
        return pet_feat * (1.0 + gamma) + beta


class CTModule(nn.Module):
    """V-Net style CT branch that provides anatomical features and optional organ logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, bottleneck=False)
        self.down_64 = DownBlock(32, 2, bottleneck=True)
        self.down_128 = DownBlock(64, 3, bottleneck=True)
        self.down_256 = DownBlock(128, 3, bottleneck=True)
        self.up_256 = UpBlock(256, 256, 3, bottleneck=True)
        self.up_128 = UpBlock(256, 128, 3, bottleneck=True)
        self.up_64 = UpBlock(128, 64, 2, bottleneck=False)
        self.up_32 = UpBlock(64, 32, 1, bottleneck=False)
        self.out_block = OutputBlock(32, out_channels)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out16 = self.in_block(x)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        return out16, out32, out64, out128, out256

    def decode(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        out16, out32, out64, out128, out256 = features
        x = self.up_256(out256, out128)
        x = self.up_128(x, out64)
        x = self.up_64(x, out32)
        x = self.up_32(x, out16)
        return self.out_block(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        features = self.encode(x)
        return {"features": features, "logits": self.decode(features)}


class EncoderModulation(nn.Module):
    """Sequential AaFM then PaFM at one encoder resolution."""

    def __init__(self, channels: int, num_experts: int):
        super().__init__()
        self.ct_film = CTConditionedFiLM(channels)
        self.experts = nn.ModuleList([nn.Conv3d(channels, channels, kernel_size=1, bias=True) for _ in range(num_experts)])
        self.eta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, pet_feat: torch.Tensor, ct_feat: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x = self.ct_film(pet_feat, ct_feat)
        mixture = torch.zeros_like(x)
        for idx, expert in enumerate(self.experts):
            w = weights[:, idx].view(weights.shape[0], 1, 1, 1, 1)
            mixture = mixture + w * expert(x)
        # PaFM is residual so expert adaptors refine rather than replace PET features.
        return x + self.eta * mixture


class PETEncoder(nn.Module):
    def __init__(self, in_channels: int, num_experts: int):
        super().__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, bottleneck=False)
        self.down_64 = DownBlock(32, 2, bottleneck=True)
        self.down_128 = DownBlock(64, 3, bottleneck=True)
        self.down_256 = DownBlock(128, 3, bottleneck=True)
        self.mod_16 = EncoderModulation(16, num_experts)
        self.mod_32 = EncoderModulation(32, num_experts)
        self.mod_64 = EncoderModulation(64, num_experts)
        self.mod_128 = EncoderModulation(128, num_experts)
        self.mod_256 = EncoderModulation(256, num_experts)

    def forward(self, x: torch.Tensor, ct_features: tuple[torch.Tensor, ...], weights: torch.Tensor) -> tuple[torch.Tensor, ...]:
        ct16, ct32, ct64, ct128, ct256 = ct_features
        pet16 = self.mod_16(self.in_block(x), ct16, weights)
        pet32 = self.mod_32(self.down_32(pet16), ct32, weights)
        pet64 = self.mod_64(self.down_64(pet32), ct64, weights)
        pet128 = self.mod_128(self.down_128(pet64), ct128, weights)
        pet256 = self.mod_256(self.down_256(pet128), ct256, weights)
        return pet16, pet32, pet64, pet128, pet256


class PETDecoder(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.up_256 = UpBlock(256, 256, 3, bottleneck=True)
        self.up_128 = UpBlock(256, 128, 3, bottleneck=True)
        self.up_64 = UpBlock(128, 64, 2, bottleneck=False)
        self.up_32 = UpBlock(64, 32, 1, bottleneck=False)
        self.out_block = OutputBlock(32, out_channels)

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        pet16, pet32, pet64, pet128, pet256 = features
        x = self.up_256(pet256, pet128)
        x = self.up_128(x, pet64)
        x = self.up_64(x, pet32)
        x = self.up_32(x, pet16)
        return self.out_block(x)


class ProMoE(nn.Module):
    """Final ProMoE network used for PET lesion segmentation."""

    def __init__(
        self,
        pet_in_channels: int = 2,
        ct_in_channels: int = 3,
        num_classes: int = 2,
        ct_num_classes: int = 33,
        text_dim: int = 34,
        num_experts: int = 12,
        temperature: float = 0.1,
        physiological_gamma: float = 2.0,
        routing_threshold: float | None = None,
    ):
        super().__init__()
        self.ct_module = CTModule(ct_in_channels, ct_num_classes)
        self.router = PrototypeRouter(
            num_experts=num_experts,
            text_dim=text_dim,
            temperature=temperature,
            physiological_gamma=physiological_gamma,
            threshold=routing_threshold,
        )
        self.pet_encoder = PETEncoder(pet_in_channels, num_experts)
        self.pet_decoder = PETDecoder(num_classes)
        self.apply(kaiming_init)
        for module in self.modules():
            if isinstance(module, CTConditionedFiLM):
                # Global initialization would otherwise overwrite FiLM identity weights.
                module.reset_to_identity()

    def forward(self, pet: torch.Tensor, ct: torch.Tensor, pe2: torch.Tensor) -> dict[str, torch.Tensor]:
        ct_out = self.ct_module(ct)
        route = self.router(pe2)
        pet_features = self.pet_encoder(pet, ct_out["features"], route["weights"])
        logits = self.pet_decoder(pet_features)
        return {
            "logits": logits,
            "prob": F.softmax(logits, dim=1),
            "ct_logits": ct_out["logits"],
            **route,
        }

    def freeze_ct(self) -> None:
        self.ct_module.eval()
        for param in self.ct_module.parameters():
            param.requires_grad_(False)

    def freeze_expert_prototypes(self) -> None:
        self.router.prototypes.requires_grad_(False)

    @property
    def max_stride(self) -> tuple[int, int, int]:
        return (16, 16, 16)
