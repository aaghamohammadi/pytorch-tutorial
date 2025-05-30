import torch
import torchvision
from torchvision import transforms


def process_mask(tensor):
    mask = tensor.squeeze(0).long()
    return mask - 1


image_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

mask_transform = transforms.Compose(
    [
        transforms.Resize(
            size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.PILToTensor(),
        transforms.Lambda(process_mask),
    ]
)


trainvalset = torchvision.datasets.OxfordIIITPet(
    root="./artifacts/data",
    split="trainval",
    download=True,
    target_types="segmentation",
    transform=image_transform,
    target_transform=mask_transform,
)
testset = torchvision.datasets.OxfordIIITPet(
    root="./artifacts/data",
    split="test",
    download=True,
    target_types="segmentation",
    transform=image_transform,
    target_transform=mask_transform,
)

print(len(trainvalset), len(testset))

print(trainvalset[0][1])
