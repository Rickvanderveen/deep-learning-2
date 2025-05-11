import click
from pathlib import Path
import csv


# do click input
@click.group()
def cli() -> None:
    pass

@cli.command("csv")
@click.option("--in_data_folder", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--out_data_folder", type=click.Path(exists=True, dir_okay=True, path_type=Path))
def convert_csv(in_data_folder, out_data_folder):
    # get all files in a directory
    all_files = in_data_folder.glob("*")
    for file in all_files:
        if file.suffix == '.csv':
            # define the output csv file paths
            output_file = out_data_folder / file.name
            # open the csv files to read from and write to
            with open(file, mode ='r') as file, open(output_file, mode='w') as file_out:
                csv_file = csv.reader(file)
                csv_file_out = csv.writer(file_out)
                # skip first row
                next(csv_file)
                # read each line in the csv file
                for lines in csv_file:
                    # change suffix to jpg to match the instagram-filtered images
                    lines[0] = (str(Path(lines[0]).with_suffix(".jpg")))
                    # write the output to the correct csv file
                    csv_file_out.writerow(lines)

if __name__ == '__main__':
    cli()