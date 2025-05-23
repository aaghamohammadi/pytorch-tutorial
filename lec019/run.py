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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./artifacts/data", train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root="./artifacts/data", train=False, download=True, transform=transform_test
)

batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


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


# Load pre-trained AlexNet
def get_pretrained_alexnet(num_classes=10):
    # Load pre-trained AlexNet
    model = torchvision.models.alexnet(weights="IMAGENET1K_V1")

    # Modify the classifier for CIFAR-10 (10 classes)
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model


clf = get_pretrained_alexnet(num_classes=10).to(device)

for param in clf.features.parameters():
    param.requires_grad = False

optimizer = optim.SGD(clf.parameters(), lr=0.0001, momentum=0.9)

criterion = nn.CrossEntropyLoss()


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


epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_epoch(trainloader, clf, criterion, optimizer)

print("Evaluating model on test data...")
evaluate(testloader, clf, criterion)
