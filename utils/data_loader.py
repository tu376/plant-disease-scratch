import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageClassificationDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None
    ):

        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        # -------------------------------------------------
        # class folders
        # -------------------------------------------------

        self.classes = sorted([
            folder_name
            for folder_name in os.listdir(root_dir)
            if os.path.isdir(
                os.path.join(root_dir, folder_name)
            )
        ])

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        # -------------------------------------------------
        # collect image paths
        # -------------------------------------------------

        for class_name in self.classes:

            class_dir = os.path.join(
                root_dir,
                class_name
            )

            class_idx = self.class_to_idx[class_name]

            for file_name in os.listdir(class_dir):

                if file_name.lower().endswith(valid_extensions):

                    image_path = os.path.join(
                        class_dir,
                        file_name
                    )

                    self.samples.append(
                        (image_path, class_idx)
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# =====================================================
# create dataloaders
# =====================================================

def create_dataloaders(
    train_dir,
    val_dir,
    batch_size=32,
    image_size=224,
    num_workers=0
):

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = ImageClassificationDataset(
        root_dir=train_dir,
        transform=train_transform
    )

    val_dataset = ImageClassificationDataset(
        root_dir=val_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (
        train_loader,
        val_loader,
        train_dataset.classes
    )


# =====================================================
# example usage
# =====================================================

if __name__ == "__main__":

    train_loader, val_loader, classes = create_dataloaders(
        train_dir="dataset/train",
        val_dir="dataset/val",
        batch_size=8,
        image_size=128
    )

    print("Classes:", classes)

    print("Number of classes:", len(classes))

    for images, labels in train_loader:

        print("Image batch shape:", images.shape)

        print("Label batch shape:", labels.shape)

        print("Labels:", labels)

        break
