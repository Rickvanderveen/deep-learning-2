import csv
import json
import argparse
from pathlib import Path


def check_images_in_csv(csv_path: Path, img_dir: Path):
    missing_imgs = []
    with csv_path.open("r", encoding="utf-8", newline="\n") as file:
        reader = csv.DictReader(file)
        for row in reader:
            img_path = row["image"]
            if not img_dir.joinpath(img_path).exists():
                missing_imgs.append(img_path)
    return missing_imgs


def main():
    parser = argparse.ArgumentParser(
        description="Check for missing images in CSV files."
    )
    parser.add_argument(
        "csv_dir",
        type=Path,
        help="Directory containing the CSV with the file locations",
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing the images.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write missing image info to missing_imgs.json",
    )

    args = parser.parse_args()
    csv_dir = Path(args.csv_dir)
    data_dir = Path(args.data_dir)

    if not csv_dir.exists():
        print("Directory for csv files does not exist")
        return
    if not data_dir.exists():
        print("Directory for the images does not exist")
        return

    print(f"Reading all csv files in directory '{csv_dir}'")
    print(f"Check if data is complete in directory '{data_dir}'")

    csv_files = csv_dir.glob("*.csv")
    missing_imgs = {
        csv_file.stem: check_images_in_csv(csv_file, data_dir) for csv_file in csv_files
    }

    if args.write:
        results_file_path = data_dir / "missing_imgs.json"
        with results_file_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(missing_imgs, file, indent=4)
        print(f"Missing images written in '{results_file_path}'")

    align_length = max(map(len, missing_imgs.keys()), default=4)

    print("File".ljust(align_length), "|", "Missing Items")
    for file, missing in missing_imgs.items():
        print(file.rjust(align_length), "|", len(missing))


if __name__ == "__main__":
    main()
