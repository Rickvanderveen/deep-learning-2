from pathlib import Path
import random
from typing import Any, Optional

import click
from tqdm import tqdm

from spai import data_utils


@click.command()
@click.option("--train_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--val_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--coco_dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--real_coco_filename", type=str, default="real_coco.txt")
@click.option("-o", "--output_csv",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True)
@click.option("-r", "--csv_root_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-d", "--output_csv_delimiter", type=str, default=",", show_default=True)
@click.option("-n", "--samples_num", type=int, default=None, show_default=True)
@click.option("-f", "--filter", type=str, multiple=True)
def main(
    train_dir: Optional[Path],
    val_dir: Optional[Path],
    coco_dir: Path,
    real_coco_filename: str,
    output_csv: Path,
    csv_root_dir: Optional[Path],
    output_csv_delimiter: str,
    samples_num: Optional[int],
    filter: list[str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = output_csv.parent

    entries: list[dict[str, Any]] = []

    coco_copy_dir_name: str = "real_coco"

    split_dirs: list[Path] = []
    split_labels: list[str] = []
    if train_dir is not None:
        split_dirs.append(train_dir)
        split_labels.append("train")
    if val_dir is not None:
        split_dirs.append(val_dir)
        split_labels.append("val")

    for s_dir, s_label in tqdm(zip(split_dirs, split_labels),
                               desc="Finding synthetic images", unit="image"):
        # Make entries for the synthetic LDM data.
        data_gen = s_dir.rglob("*")
        for p in data_gen:
            # if filetype.is_image(p):
            path_parts: list[str] = p.parts
            if (p.is_file() and p.suffix == ".png"
                    and coco_copy_dir_name not in path_parts):
                filter_found: bool = False if len(filter) > 0 else True
                for f in filter:
                    if f in path_parts:
                        filter_found = True
                        break
                if not filter_found:
                    continue

                entries.append({
                    "image": str(p.relative_to(csv_root_dir)),
                    "class": 1,
                    "split": s_label
                })
                # valid_num += 1

        # Make entries for COCO real data.
        real_coco_file: Path = s_dir / real_coco_filename
        coco_samples: list[Path] = find_coco_samples(real_coco_file, coco_dir, s_label)
        for p in coco_samples:
            entries.append({
                "image": str(p.relative_to(csv_root_dir)),
                "class": 0,
                "split": s_label
            })

    if samples_num is not None:
        entries = random.sample(entries, samples_num)

    data_utils.write_csv_file(entries, output_csv, delimiter=output_csv_delimiter)
    print(f"Exported CSV to {output_csv}")


def find_coco_samples(coco_real_file: Path, coco_dir: Path, split: str) -> list[Path]:
    assert split in ["train", "val"]
    print("Loading COCO image paths.")
    with coco_real_file.open() as f:
        lines: list[str] = [l.rstrip() for l in f]
    coco_files: list[Path] = [coco_dir / f"train2017" / l for l in lines]
    print("Loading of COCO image paths completed.")
    for f in tqdm(coco_files, "Checking existence of COCO images", unit="image"):
        if not f.exists():
            continue
    return coco_files


if __name__ == "__main__":
    main()
