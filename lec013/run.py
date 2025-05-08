import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

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

batch_size = 4

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


class CIFAR10Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
