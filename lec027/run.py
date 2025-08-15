import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bridge = DoubleConv(128, 256)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(256, 128)  # 256 = 128 + 128 (skip connection)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)  # 128 = 64 + 64 (skip connection)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(64, 32)  # 64 = 32 + 32 (skip connection)

        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(32, 16)  # 32 = 16 + 16 (skip connection)

        # Output layer
        self.outconv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)

        enc2 = self.encoder2(x)
        x = self.pool2(enc2)

        enc3 = self.encoder3(x)
        x = self.pool3(enc3)

        enc4 = self.encoder4(x)
        x = self.pool4(enc4)

        # Bridge
        x = self.bridge(x)

        # Decoder with skip connections
        x = self.upconv1(x)
        x = torch.cat([x, enc4], dim=1)  # Skip connection (dim=1 for channels)
        x = self.decoder1(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder2(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder3(x)

        x = self.upconv4(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder4(x)

        # Output layer
        x = self.outconv(x)
        return x


# Initialize model
model = UNet(in_channels=3, out_channels=3)  # 3 classes for segmentation

# Move model to device
model = model.to(device)

model = torch.compile(model)


def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0.0

    for _, (images, masks) in enumerate(dataloader):
        # Move data to device
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average loss
    train_loss /= len(dataloader)
    print(f"Train avg loss: {train_loss:>8f}")

    return train_loss


def evaluate(dataloader, model, loss_fn):
    model.eval()
    val_loss = 0.0
    pixel_correct = 0
    pixel_total = 0

    # For IoU calculation
    intersection_sum = 0
    union_sum = 0

    with torch.no_grad():
        for _, (images, masks) in enumerate(dataloader):
            # Move data to device
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Calculate pixel accuracy
            pixel_correct += (preds == masks).sum().item()
            pixel_total += masks.numel()

            # Calculate IoU
            for cls in range(3):  # 3 classes in your case
                pred_inds = preds == cls
                target_inds = masks == cls
                intersection = (pred_inds & target_inds).sum().item()
                union = (pred_inds | target_inds).sum().item()

                if union > 0:
                    intersection_sum += intersection
                    union_sum += union

    # Calculate metrics
    val_loss /= len(dataloader)
    pixel_accuracy = 100.0 * pixel_correct / pixel_total
    mean_iou = 100.0 * intersection_sum / union_sum if union_sum > 0 else 0

    print(f"Validation loss: {val_loss:.6f}")
    print(f"Pixel accuracy: {pixel_accuracy:.2f}%")
    print(f"Mean IoU: {mean_iou:.2f}%")

    return val_loss, pixel_accuracy, mean_iou


batch_size = 8
train_dataloader = DataLoader(trainvalset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


num_epochs = 20


print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_epoch(train_dataloader, model, criterion, optimizer)
    evaluate(test_dataloader, model, criterion)

print("Training complete!")
