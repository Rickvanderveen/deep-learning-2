from PIL import Image
import pilgram2
import random
import click
from pathlib import Path
import csv


# do click input
@click.group()
def cli() -> None:
    pass

@cli.command("trainset")
@click.option("--input_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def training_images(input_csv):
    # open csv file
    with open(input_csv, mode ='r') as file:
        csv_file = csv.reader(file)
        # skip first row
        next(csv_file)
        # read each line in the csv file
        for lines in csv_file:
            # read the original image
            img = Image.open('./datasets/' + lines[0], mode='r')
            # get the path where the modified images should be written to
            path_images = Path('./reduced_trainingset/' + lines[0])
            path_parents = path_images.parent
            # if the parent directories did not exist before, make new (nested) directories
            if not path_parents.exists():
                path_parents.mkdir(parents=True, exist_ok=True)
            # save the image with Instagram-like filters and apply jpeg commpression
            img.save(path_images)

if __name__ == '__main__':
    cli()