import matplotlib.pyplot as plt
import numpy as np
import torchvision

trainset = torchvision.datasets.CIFAR10(
    root="./artifacts/data", train=True, download=True
)
testset = torchvision.datasets.CIFAR10(
    root="./artifacts/data", train=False, download=True
)
print(trainset[0])

image, label = trainset[0]

np_image = np.array(image)
print(np_image.shape)

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

plt.imshow(image)
plt.title(f"{class_names[label]} ({label})")
plt.axis("off")
plt.savefig("artifacts/frog.png")
