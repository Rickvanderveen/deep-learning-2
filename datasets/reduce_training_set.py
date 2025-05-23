from pathlib import Path
import csv

input_file = "ldm_train_val.csv"
output_file = "reduced_training_data.csv"

with open(input_file, mode ='r') as file, open(output_file, mode='w') as file_out:
    csv_file = csv.reader(file)
    csv_file_out = csv.writer(file_out)
    # write header
    header = next(csv_file)
    csv_file_out.writerow(header)
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0
    val_count1 = 0
    val_count2 = 0
    val_count3 = 0
    val_count4 = 0
    val_count5 = 0
    val_count6 = 0
    val_count7 = 0
    val_count8 = 0
    # read each line in the csv file
    for lines in csv_file:
        # get 16198 train images from coco and latent_diffusion_trainingset
        if lines[2] == 'train' and 'train/latent_diffusion_text2img_set1' in lines[0] and count1 < 2314:
            csv_file_out.writerow(lines)
            count1 += 1
        elif lines[2] == 'train' and 'train/latent_diffusion_noise2image_bedrooms' in lines[0] and count2 < 2314:
            csv_file_out.writerow(lines)
            count2 += 1
        elif lines[2] == 'train' and 'train/latent_diffusion_text2img_set2' in lines[0] and count3 < 2314:
            csv_file_out.writerow(lines)
            count3 += 1
        elif lines[2] == 'train' and 'train/latent_diffusion_class2image' in lines[0] and count4 < 2314:
            csv_file_out.writerow(lines)
            count4 += 1      
        elif lines[2] == 'train' and 'train/latent_diffusion_noise2image_FFHQ' in lines[0] and count5 < 2314:
            csv_file_out.writerow(lines)
            count5 += 1
        elif lines[2] == 'train' and 'train/latent_diffusion_text2img_set0' in lines[0] and count6 < 2314:
            csv_file_out.writerow(lines)
            count6 += 1
        elif lines[2] == 'train' and 'train/latent_diffusion_noise2image_churches' in lines[0] and count7 < 2314:
            csv_file_out.writerow(lines)
            count7 += 1
        elif lines[2] == 'train' and 'COCO/train2017' in lines[0] and count8 < 16198:
            csv_file_out.writerow(lines)
            count8 += 1    

        # get 1799 validation images from coco and latent_diffusion_trainingset
        if lines[2] == 'val' and 'valid/latent_diffusion_text2img_set1' in lines[0] and val_count1 < 257:
            csv_file_out.writerow(lines)
            val_count1 += 1
        elif lines[2] == 'val' and 'valid/latent_diffusion_noise2image_bedrooms' in lines[0] and val_count2 < 257:
            csv_file_out.writerow(lines)
            val_count2 += 1
        elif lines[2] == 'val' and 'valid/latent_diffusion_text2img_set2' in lines[0] and val_count3 < 257:
            csv_file_out.writerow(lines)
            val_count3 += 1
        elif lines[2] == 'val' and 'valid/latent_diffusion_class2image' in lines[0] and val_count4 < 257:
            csv_file_out.writerow(lines)
            val_count4 += 1      
        elif lines[2] == 'val' and 'valid/latent_diffusion_noise2image_FFHQ' in lines[0] and val_count5 < 257:
            csv_file_out.writerow(lines)
            val_count5 += 1
        elif lines[2] == 'val' and 'valid/latent_diffusion_text2img_set0' in lines[0] and val_count6 < 257:
            csv_file_out.writerow(lines)
            val_count6 += 1
        elif lines[2] == 'val' and 'valid/latent_diffusion_noise2image_churches' in lines[0] and val_count7 < 257:
            csv_file_out.writerow(lines)
            val_count7 += 1
        elif lines[2] == 'val' and 'COCO/train2017' in lines[0] and val_count8 < 1799:
            csv_file_out.writerow(lines)
            val_count8 += 1  