import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./artifacts/data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./artifacts/data", train=False, download=True, transform=transform
)
print(trainset[0])

image, label = trainset[0]

print(image.shape)

# Class names in CIFAR-10
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
