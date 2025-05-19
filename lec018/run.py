import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


transform_train = transforms.Compose(
    [
        transforms.Resize(227),  # Resize to 227x227
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


transform_test = transforms.Compose(
    [
        transforms.Resize(227),  # Resize to 227x227
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./artifacts/data", train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root="./artifacts/data", train=False, download=True, transform=transform_test
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


def imshow(img):
    # img is a PyTorch tensor with shape (C,H,W) and normalized values
    mean = 0.5
    std = 0.5
    img = std * img + mean  # Denormalizes the image
    img = img.numpy().transpose((1, 2, 0))  # Changes from CHW to HWC format
    plt.imshow(img)
    plt.show()


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


clf = AlexNet().to(device)

batch_size = 4

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(clf.parameters(), lr=0.001, momentum=0.9)


def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0.0
    for _, (X, y) in enumerate(dataloader):
        # Move data to device
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
    train_loss /= len(dataloader)
    print(f"Train avg loss: {train_loss:>8f}")


def evaluate(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            # Move data to device
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    test_loss /= len(dataloader)
    accuracy = 100 * correct / total
    print(f"Test avg loss: {test_loss:>8f}, Accuracy: {accuracy:>0.1f}%")
    return test_loss, accuracy


# Training loop
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_epoch(trainloader, clf, criterion, optimizer)

# Add evaluation after training
print("Evaluating model on test data...")
evaluate(testloader, clf, criterion)
