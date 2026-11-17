"""Target classifiers: ImageNet-pretrained torchvision models with a one-channel input layer.

The RGB weights of the first convolution are averaged into one input channel, and the last layer is replaced
by a num_classes head. Inputs are grayscale 224 x 224 images normalised with mean 0.5 / std 0.5.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (ResNet18_Weights, ResNet50_Weights, VGG16_Weights, ViT_B_16_Weights,
                                ViT_B_32_Weights)

MODELS = ("resnet18", "resnet50", "vgg16", "vit_b_16", "vit_b_32")
INPUT_SIZE = 224


def one_channel_conv(conv: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(1, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None,
                    padding_mode=conv.padding_mode)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_model(name, num_classes, pretrained=True):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == "resnet50":
        m = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    elif name in ("vgg16", "vgg"):
        m = models.vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
        m.features[0] = one_channel_conv(m.features[0])
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        return m
    elif name in ("vit_b_16", "vit_b_32"):
        ctor, w = (models.vit_b_16, ViT_B_16_Weights) if name == "vit_b_16" else (models.vit_b_32, ViT_B_32_Weights)
        m = ctor(weights=w.DEFAULT if pretrained else None)
        m.conv_proj = one_channel_conv(m.conv_proj)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
    m.conv1 = one_channel_conv(m.conv1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "network"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if checkpoint and all(k.startswith("module.") for k in checkpoint):
        checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}
    return checkpoint


def load_classifier(name, num_classes, checkpoint, device="cuda"):
    """A trained classifier in eval mode with frozen weights."""
    m = build_model(name, num_classes, pretrained=False)
    state = _state_dict(torch.load(Path(checkpoint), map_location=device))
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{checkpoint} is missing parameters: {missing[:10]}")
    if unexpected:
        print("WARNING: unexpected keys in", checkpoint, unexpected[:10])
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def classifier_transform(size=INPUT_SIZE):
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5])])


def checkpoint_path(root, dataset, model):
    """Default location of a trained classifier: <root>/checkpoints/<dataset>/classifiers/<model>/final_model.pt."""
    return Path(root) / "checkpoints" / dataset / "classifiers" / model / "final_model.pt"
