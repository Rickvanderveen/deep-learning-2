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

@cli.command("filter")
@click.option("--input_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def filtered_images(input_csv):
    # available Instagram-like filters from pilgram2
    filters = [pilgram2._1977, pilgram2.aden, pilgram2.ashby, pilgram2.amaro, pilgram2.brannan, pilgram2.brooklyn, pilgram2.charmes, 
            pilgram2.clarendon, pilgram2.crema, pilgram2.dogpatch, pilgram2.earlybird, pilgram2.gingham, pilgram2.ginza, pilgram2.hefe, 
            pilgram2.helena, pilgram2.hudson, pilgram2.inkwell, pilgram2.juno, pilgram2.kelvin, pilgram2.lark, pilgram2.lofi, pilgram2.ludwig, 
            pilgram2.maven, pilgram2.mayfair, pilgram2.moon, pilgram2.nashville, pilgram2.perpetua, pilgram2.poprocket, pilgram2.reyes, 
            pilgram2.rise, pilgram2.sierra, pilgram2.skyline, pilgram2.slumber, pilgram2.stinson, pilgram2.sutro, pilgram2.toaster, 
            pilgram2.valencia, pilgram2.walden, pilgram2.willow, pilgram2.xpro2]
    
    # open csv file
    with open(input_csv, mode ='r') as file:
        csv_file = csv.reader(file)
        # skip first row
        next(csv_file)
        # read each line in the csv file
        for lines in csv_file:
            # read the original image
            img = Image.open('./data/' + lines[0], mode='r')
            # randomly choose a filter
            random_choice_filter = random.choice(filters)
            # print(random_choice_filter.__name__)
            # apply the filter
            filter_on_img = random_choice_filter(img)
            # get the path where the modified images should be written to
            path_images = Path('./data_with_filter/' + lines[0])
            path_parents = path_images.parent
            # if the parent directories did not exist before, make new (nested) directories
            if not path_parents.exists():
                path_parents.mkdir(parents=True, exist_ok=True)
            # save the image with Instagram-like filters and apply jpeg commpression
            path_images = path_images.with_suffix(".jpg")
            filter_on_img.save(path_images, format="JPEG", quality=75)

if __name__ == '__main__':
    cli()
