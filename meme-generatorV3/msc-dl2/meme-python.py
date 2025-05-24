import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap
import re
from pathlib import Path
import csv

def scramble_word(word):
    if len(word) <= 3:
        return word
    middle = list(word[1:-1])
    random.shuffle(middle)
    return word[0] + ''.join(middle) + word[-1]

def scramble_sentence(sentence):
    words = re.findall(r'\w+|\W+', sentence)  # Keep punctuation separate
    scrambled = [scramble_word(word) if word.isalpha() else word for word in words]
    return ''.join(scrambled)

def generate_caption_from_textfile(text_file="genesis.txt", min_len=20, max_len=100):
    try:
        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read().replace('\n', ' ').strip()

        # Remove verse numbers like "01:001:001"
        content = re.sub(r'\b\d{2}:\d{3}:\d{3}\b', '', content)

        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= min_len]
        if not sentences:
            raise ValueError("No valid sentence found.")

        selected = random.choice(sentences)
        scrambled = scramble_sentence(selected)
        print("Generated Scrambled Caption:", scrambled)
        return scrambled

    except Exception as e:
        print(f"Error reading or processing caption: {e}")
        return "TEXT NOT FOUND"

def add_meme_text(image, text, max_width_ratio=0.9, min_spacing=20):
    image = image.convert("RGBA")
    width, height = image.size

    font_size = random.randint(int(height * 0.05), int(height * 0.10))
    font_type = random.randint(0, 2)

    if font_type == 0:
        font = ImageFont.truetype("impact.ttf", size=font_size)
    elif font_type == 1:
        font = ImageFont.truetype("Comic Sans MS.ttf", size=font_size)
    elif font_type == 2:
        font = ImageFont.truetype("ARIAL.TTF", size=font_size)

    text = text.upper()
    max_text_width = width * max_width_ratio
    rough_wrap = wrap(text, width=30)
    wrapped_lines = []

    text_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_overlay)

    for line in rough_wrap:
        words = line.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    wrapped_lines.append(current_line)
                current_line = word
        if current_line:
            wrapped_lines.append(current_line)

    bbox = draw.textbbox((0, 0), "A", font=font)
    line_height = (bbox[3] - bbox[1]) + 5

    half = len(wrapped_lines) // 2
    top_lines = wrapped_lines[:half] if half > 0 else wrapped_lines
    bottom_lines = wrapped_lines[half:] if half > 0 else []

    max_offset = int(height * 0.04)
    vertical_offset = random.randint(-max_offset, max_offset)

    def draw_text_lines(lines, y_start):
        y = y_start + vertical_offset
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            max_x = width - text_width
            x = random.randint(0, max(0, max_x))

            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
            y += line_height

    draw_text_lines(top_lines, min_spacing)

    total_bottom_height = len(bottom_lines) * line_height
    total_top_height = len(top_lines) * line_height
    y_start_bottom = max(total_top_height + min_spacing - 10,
                         height - total_bottom_height - min_spacing - 20)
    draw_text_lines(bottom_lines, y_start_bottom)

    image = Image.alpha_composite(image, text_overlay)
    return image



def add_random_emojis(image, min_count=5, max_count=10, emoji_folder="emojis"):
    width, height = image.size
    EMOJIS = [
        'clown_face.png', 'fire.png', 'heart_eyes.png', 'joy.png',
        'pensive.png', 'rolling_on_the_floor_laughing.png',
        'skull.png', 'sunglasses.png', 'tada.png', 'thinking_face.png'
    ]
    count = random.randint(min_count, max_count)

    for _ in range(count):
        emoji_size = random.randint(int(height * 0.04), int(height * 0.16))
        emoji_file = random.choice(EMOJIS)
        emoji_path = os.path.join(emoji_folder, emoji_file)
        emoji_img = Image.open(emoji_path).convert("RGBA")
        emoji_img = emoji_img.resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)
        x = random.randint(0, width - emoji_size)
        y = random.randint(0, height - emoji_size)
        image.paste(emoji_img, (x, y), emoji_img)
    return image

def process_image(input_path, output_path):
    try:
        image = Image.open(input_path).convert("RGBA")
        caption = generate_caption_from_textfile("genesis_cleaned.txt")
        meme = add_meme_text(image, caption)
        meme_with_emojis = add_random_emojis(meme)
        meme_with_emojis.save(output_path)
        print(f"✅ Saved meme: {output_path}")
    except Exception as e:
        print(f"❌ Failed on {input_path}: {e}")

def batch_process_images(input_csv):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # for filename in os.listdir(input_dir):
    #     if filename.lower().endswith(supported_formats):
    #         input_path = os.path.join(input_dir, filename)
    #         output_filename = os.path.splitext(filename)[0] + ".png"
    #         output_path = os.path.join(output_dir, output_filename)
    #         process_image(input_path, output_path)
    
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    with open(input_csv, mode ='r') as file:
        csv_file = csv.reader(file)
        # skip first row
        next(csv_file)
        # read each line in the csv file
        for lines in csv_file:
            if lines[0].endswith(supported_formats):
                # get the path where the modified images should be written to
                input_path = Path('./home/deep-learning-2/data/' + lines[0])
                path_images = Path('./data_with_memes/' + lines[0])
                path_parents = path_images.parent
                # if the parent directories did not exist before, make new (nested) directories
                if not path_parents.exists():
                    path_parents.mkdir(parents=True, exist_ok=True)
                # save the image with Instagram-like filters and apply jpeg commpression
                path_images = path_images.with_suffix(".png")
                output_path = path_images
                # print(input_path)
                process_image(input_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python meme_generator.py <output_csv>")
        sys.exit(1)

    # input_folder = sys.argv[1]
    input_csv = sys.argv[1]


    batch_process_images(input_csv)
