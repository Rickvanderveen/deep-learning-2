import csv
from pathlib import Path

import click
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import default_collate
from tqdm import tqdm

from BSRGAN.models.network_rrdbnet import RRDBNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageDataset(Dataset):
    def __init__(self, csv_path: Path, root_dir: Path = Path(), scale_factor: int = 1):
        """
        Args:
            csv_file (Path): Path to the CSV file.
            root_dir (Path): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir

        self.pil_to_torch = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame["image"].iloc[idx]
        img_path = self.root_dir / img_name
        img = Image.open(img_path, mode="r").convert("RGB")
        tensor_img = self.pil_to_torch(img)
        _, height, width = tensor_img.shape

        tensor_img = transforms.functional.resize(
            tensor_img, [height // self.scale_factor, width // self.scale_factor]
        )

        label = self.data_frame["class"].iloc[idx]
        split = self.data_frame["split"].iloc[idx]

        return {
            "image": tensor_img,
            "image_name": img_name,
            "label": label,
            "split": split,
        }


@torch.inference_mode()
def single_super_resolution(
    data_loader: DataLoader,
    output_dir: Path,
    model: torch.nn.Module,
    output_csv_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToPILImage(mode="RGB")])

    output_csv = output_dir / output_csv_name
    with output_csv.open(mode="w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["image", "class", "split"])

        for batch in tqdm(data_loader, total=len(data_loader)):
            images = batch["image"].to(device)
            image_names = batch["image_name"]
            labels = batch["label"]
            splits = batch["split"]

            # Apply super resolution
            sr_tensors = model(images)
            sr_tensors = torch.clamp(sr_tensors, 0.0, 1.0)

            for img_idx, tensor in enumerate(sr_tensors):
                output_image = transform(tensor)
                image_name = image_names[img_idx]
                # get the path where the modified images should be written to
                output_image_path = output_dir / image_name
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                output_image.save(output_image_path)

                # Update the output csv with the new image
                writer.writerow([image_name, labels[img_idx], splits[img_idx]])


@click.command()
@click.option(
    "--input_csv",
    "-f",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="One or multiple input csv. Should have the same parent directory",
)
@click.option(
    "--root_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(),
    help="The directory of the input_csv(s) and the images",
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="The output directory where the csv and images will be placed",
)
@click.option(
    "--batch_size",
    "-b",
    type=int,
    default=1,
    help="The number of images that will be in a single batch",
)
@click.option(
    "--scale_factor",
    type=int,
    default=2,
    help="The factor of the downscale and upscale by SR.",
)
def multiple_super_resolution(
    input_csv: tuple[Path],
    root_dir: Path,
    output_dir: Path,
    batch_size: int = 1,
    scale_factor: int = 2,
):
    all_same_parent = (
        all(map(lambda path: path == input_csv[0], input_csv)) and len(input_csv) > 1
    )
    if all_same_parent:
        raise ValueError(
            "The parent directory should be the same for multiple input csv"
        )

    if scale_factor == 2:
        model_name = "BSRGANx2"
    elif scale_factor == 4:
        model_name = "BSRGAN"
    else:
        raise ValueError(f"Scale factor of {scale_factor} is not supported. Use 2 or 4")

    model_path = Path("BSRGAN", "model_zoo", f"{model_name}.pth")

    torch.cuda.empty_cache()
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale_factor)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.to(device)
    model.eval()

    def shape_checking_collate_fn(batch):
        first_shape = batch[0]["image"].shape
        for i, item_ in enumerate(batch):
            item = item_["image"]
            if item.shape != first_shape:
                raise ValueError(
                    f"Item {i} ({item_['image_name']}) has shape {item.shape}, expected {first_shape}"
                )

        return default_collate(batch)  # Proceed with regular batching

    for input_file in input_csv:
        print(f"Using {model_name} SR on images in {input_file.name}")

        dataset = ImageDataset(
            input_file,
            root_dir,
            scale_factor,
        )
        data_loader = DataLoader(
            dataset,
            batch_size,
            collate_fn=shape_checking_collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        single_super_resolution(data_loader, output_dir, model, input_file.name)


if __name__ == "__main__":
    multiple_super_resolution()
