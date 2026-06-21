from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import kagglehub
from PIL import Image, ImageOps
from tqdm import tqdm


DATASET_HANDLE = "abdallahalidev/plantvillage-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PlantVillage data.")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing output_dir before preparing the dataset.",
    )
    return parser.parse_args()

from PIL import ImageEnhance

def augment_image(img):
    ops = [
        lambda x: x.rotate(random.uniform(-20, 20)),
        lambda x: ImageOps.mirror(x),
        lambda x: ImageEnhance.Brightness(x).enhance(
            random.uniform(0.8, 1.2)
        ),
        lambda x: ImageEnhance.Contrast(x).enhance(
            random.uniform(0.8, 1.2)
        ),
    ]

    op = random.choice(ops)
    return op(img)
def find_color_dataset_root(download_dir: Path) -> Path:
    """Find the PlantVillage color image folder returned by kagglehub."""
    candidates = [
        download_dir / "plantvillage dataset" / "color",
        download_dir / "PlantVillage" / "color",
        download_dir / "color",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
            return candidate

    for candidate in download_dir.rglob("color"):
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
            return candidate

    raise FileNotFoundError(
        f"Could not find PlantVillage color folder inside {download_dir}"
    )


def class_slug(class_name: str) -> str:
    return (
        class_name.lower()
        .replace("___", "_")
        .replace("__", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )


def image_files(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_resized_image(src: Path, dst: Path, image_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image.save(dst, format="JPEG", quality=95)


TARGET_TOTAL = 625
TRAIN_COUNT = 500
VAL_COUNT = 62
TEST_COUNT = 63


def prepare_class(
    class_dir: Path,
    output_dir: Path,
    image_size: int,
    rng: random.Random,
):
    files = image_files(class_dir)

    if len(files) < 10:
        raise ValueError(
            f"Class {class_dir.name} has too few images: {len(files)}"
        )

    rng.shuffle(files)

    # 80/10/10 split on ORIGINAL images
    n = len(files)

    train_orig = files[: int(0.8 * n)]
    val_files = files[int(0.8 * n): int(0.9 * n)]
    test_files = files[int(0.9 * n):]
    train_samples = [
        ("orig", f, 0)
        for f in train_orig
    ]

    # If too many train images, subsample
    if len(train_samples) > TRAIN_COUNT:
        train_samples = rng.sample(
            train_samples,
            TRAIN_COUNT
        )

    # If too few train images, augment
    while len(train_samples) < TRAIN_COUNT:

        src = rng.choice(train_orig)

        train_samples.append(
            (
                "aug",
                src,
                len(train_samples)
            )
        )

    val_samples = [
        ("orig", f, 0)
        for f in val_files
    ]

    if len(val_samples) > VAL_COUNT:
        val_samples = rng.sample(
            val_samples,
            VAL_COUNT
        )

    test_samples = [
        ("orig", f, 0)
        for f in test_files
    ]

    if len(test_samples) > TEST_COUNT:
        test_samples = rng.sample(
            test_samples,
            TEST_COUNT
        )

    slug = class_slug(class_dir.name)

    counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    for split, subset in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples)
    ]:

        for mode, src, idx in tqdm(
            subset,
            desc=f"{slug}-{split}",
            leave=False
        ):

            dst = (
                output_dir
                / split
                / slug
                / f"{src.stem}_{idx}.jpg"
            )

            dst.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with Image.open(src) as image:

                image = (
                    ImageOps
                    .exif_transpose(image)
                    .convert("RGB")
                )

                if mode == "aug":
                    image = augment_image(image)

                image = image.resize(
                    (image_size, image_size),
                    Image.Resampling.LANCZOS
                )

                image.save(
                    dst,
                    format="JPEG",
                    quality=95
                )

            counts[split] += 1

    return (
        counts["train"],
        counts["val"],
        counts["test"]
    )

def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val_ratio must be between 0 and 1")

    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)

    print("Downloading PlantVillage dataset with kagglehub...")
    download_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    source_root = find_color_dataset_root(download_path)

    class_dirs = sorted(p for p in source_root.iterdir() if p.is_dir())
    selected_classes = class_dirs[:20]
    if len(selected_classes) < args.num_classes:
        raise ValueError(
            f"Requested {args.num_classes} classes, found only {len(selected_classes)}"
        )

    print(f"Source: {source_root}")
    print("Selected classes:")
    for class_dir in selected_classes:
        print(f"  - {class_dir.name} -> {class_slug(class_dir.name)}")

    rng = random.Random(args.seed)
    totals: list[tuple[str, int, int]] = []
    for class_dir in selected_classes:

        train_count, val_count, test_count = prepare_class(
            class_dir=class_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
            rng=rng,
        )

        totals.append(
            (
                class_slug(class_dir.name),
                train_count,
                val_count,
                test_count,
            )
        )

    print("\nPrepared dataset:")
    for slug, train_count, val_count, test_count in totals:
        print(
            f"{slug}: "
            f"train={train_count}, "
            f"val={val_count}, "
            f"test={test_count}"
        )


if __name__ == "__main__":
    main()
