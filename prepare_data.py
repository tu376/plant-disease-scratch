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


def prepare_class(
    class_dir: Path,
    output_dir: Path,
    image_size: int,
    val_ratio: float,
    rng: random.Random,
) -> tuple[int, int]:
    files = image_files(class_dir)
    if not files:
        raise ValueError(f"No image files found in {class_dir}")

    shuffled = files[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_files = set(shuffled[:val_count])
    train_count = 0
    val_saved_count = 0
    slug = class_slug(class_dir.name)

    for src in tqdm(shuffled, desc=slug, leave=False):
        split = "val" if src in val_files else "train"
        dst = output_dir / split / slug / f"{src.stem}.jpg"
        save_resized_image(src, dst, image_size)
        if split == "train":
            train_count += 1
        else:
            val_saved_count += 1

    return train_count, val_saved_count


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
    selected_classes = class_dirs[: args.num_classes]
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
        train_count, val_count = prepare_class(
            class_dir=class_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        totals.append((class_slug(class_dir.name), train_count, val_count))

    print("\nPrepared dataset:")
    for slug, train_count, val_count in totals:
        print(f"  {slug}: train={train_count}, val={val_count}")
    print(f"\nDone. Files saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
