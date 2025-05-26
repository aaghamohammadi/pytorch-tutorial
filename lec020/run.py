import matplotlib.pyplot as plt
import torchvision


def display_image_with_mask(img, mask, save_path=None):
    """
    Display an image and its segmentation mask side by side.

    Parameters:
    -----------
    img : PIL.Image
        The input image
    mask : PIL.Image
        The segmentation mask
    save_path : str, optional
        Path to save the figure
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the image
    ax1.imshow(img)
    ax1.set_title("Pet Image")
    ax1.axis("off")

    # Display the segmentation mask
    ax2.imshow(mask, cmap="gray")
    ax2.set_title("Segmentation Mask")
    ax2.axis("off")

    plt.tight_layout()

    plt.savefig(save_path)


trainvalset = torchvision.datasets.OxfordIIITPet(
    root="./artifacts/data",
    split="trainval",
    download=True,
    target_types="segmentation",
)
testset = torchvision.datasets.OxfordIIITPet(
    root="./artifacts/data", split="test", download=True, target_types="segmentation"
)

print(len(trainvalset), len(testset))

# Get the first sample
img, mask = trainvalset[0]

# Display the image and mask
display_image_with_mask(img, mask, "artifacts/pet_segmentation.png")
