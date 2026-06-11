import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cupy as cp

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
    test_dir,
    batch_size=32,
    image_size=64,
    num_workers=0
):

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageClassificationDataset(
        root_dir=train_dir,
        transform=train_transform
    )
    test_dataset = ImageClassificationDataset(
        root_dir=test_dir,
        transform=test_transform
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )
def dataloader_to_cupy(loader):

    X_batches = []
    y_batches = []

    for images, labels in loader:

        X_batches.append(cp.asarray(images))
        y_batches.append(cp.asarray(labels))

    X = cp.concatenate(X_batches, axis=0)
    y = cp.concatenate(y_batches, axis=0)

    return X, y