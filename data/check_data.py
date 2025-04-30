import csv
import json
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
    file_path = Path(__file__)
    data_dir = file_path.parent
    csv_files = data_dir.glob("*.csv")

    missing_imgs = {csv_file.stem: check_images_in_csv(csv_file, data_dir) for csv_file in csv_files}
    
    results_file_path = data_dir.joinpath("missing_imgs.json")
    with results_file_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(missing_imgs, file, indent=4)
    
    align_length = max(map(lambda x: len(x), missing_imgs.keys()))

    print("File".ljust(align_length, " "), "|", "Missing Items")
    for file, missing in missing_imgs.items():
        print(file.rjust(align_length, " "), "|", len(missing))

if __name__ == "__main__":
    main()
