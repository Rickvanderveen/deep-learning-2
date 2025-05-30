from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from jinja2 import Template
import random
import os
import string
import base64
import argparse

def choose_random_value(arr):
    return random.choice(arr)

def get_random_timestamp():
    units = ["minutes", "hours", "days", "weeks", "months"]
    unit = random.choice(units)
    number = random.randint(1, 99)
    return f"{number} {unit} ago"

def generate_random_caption(min_length=10, max_length=30):
    length = random.randint(min_length, max_length)
    words = []
    while len(' '.join(words)) < length:
        word_len = random.randint(2, 8)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
    caption = ' '.join(words)
    return caption[:length].strip()

def get_random_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{hour}:{minute:02d}"

def generate_name():
    vowels = 'aeiou'
    consonants = ''.join(set(string.ascii_lowercase) - set(vowels))
    
    def make_name(min_len=4, max_len=8):
        length = random.randint(min_len, max_len)
        name = ''
        for i in range(length):
            if i % 2 == 0:
                name += random.choice(consonants)
            else:
                name += random.choice(vowels)
        return name.capitalize()
    
    return f"{make_name()} {make_name()}"

def generate_instagram_string():
    value = random.uniform(0, 2_000_000)

    if value < 1_000:
        return str(int(value))
    elif value < 1_000_000:
        return f"{value / 1_000:.1f}K".rstrip('0').rstrip('.')
    else:
        return f"{value / 1_000_000:.1f}M".rstrip('0').rstrip('.')

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
    
def svg_to_base64(path):
    with open(path, "rb") as svg_file:
        encoded = base64.b64encode(svg_file.read()).decode('utf-8')
        return f"data:image/svg+xml;base64,{encoded}"



def generate_post(driver, image_path, profile_picture, output_path):
    with open("template.html") as f:
        html_template = Template(f.read())

    filled_html = html_template.render(
        username=generate_name(),
        caption=generate_random_caption(8, 20),
        like_count=generate_instagram_string(),
        comment_count=generate_instagram_string(),
        share_count=generate_instagram_string(),
        post_image=image_to_base64(os.path.abspath(image_path)),
        profile_picture=image_to_base64(os.path.abspath(profile_picture)),
        timestamp=get_random_timestamp(),
        heart_icon=svg_to_base64(choose_random_value(["assets/icons/heart_notif.svg", "assets/icons/heart.svg"])),
        message_icon=svg_to_base64(choose_random_value(["assets/icons/send-message.svg", "assets/icons/send-message-notif.svg"])),
        save_icon=svg_to_base64(choose_random_value(["assets/icons/save.svg", "assets/icons/save-filled.svg"])),
        battery_icon=svg_to_base64(choose_random_value(["assets/icons/battery-empty.svg", "assets/icons/battery-full.svg"])),
        cellular_icon=svg_to_base64(choose_random_value(["assets/icons/cellular-1.svg", "assets/icons/cellular-2.svg", "assets/icons/cellular-3.svg", "assets/icons/cellular-4.svg", "assets/icons/cellular-5.svg"])),
        network_icon=svg_to_base64(choose_random_value(["assets/icons/network-3g.svg", "assets/icons/network-4g.svg"])),
        system_time=get_random_time(),
        home_icon=svg_to_base64("assets/icons/home.svg"),
        search_icon=svg_to_base64("assets/icons/search.svg"),
        plus_square=svg_to_base64("assets/icons/plus-square.svg"),
        movie_icon=svg_to_base64("assets/icons/movie.svg"),
        heart_post_icon=svg_to_base64("assets/icons/heart.svg"),
        chat_icon=svg_to_base64("assets/icons/chatbubble.svg"),
        direct_message_icon=svg_to_base64("assets/icons/send-message.svg"),
        down_arrow=svg_to_base64("assets/icons/down-arrow.svg")
    )

    # Load the information onto the page
    driver.get("about:blank")
    driver.execute_script("document.write(arguments[0]);", filled_html)
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".nav-avatar"))
    )

    # Get the element we want
    post_container = driver.find_element(By.CSS_SELECTOR, '.post-container')
    driver.execute_script("arguments[0].scrollIntoView(true);", post_container) # Make sure we look at the full element
    post_container.screenshot(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Input and output of instagram simulation")
    parser.add_argument("--input_dir", type=str, default="./assets/imgs")
    parser.add_argument("--avatars_dir", type=str, default="./assets/avatars")
    parser.add_argument("--output_dir", type=str, default="./output")

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    params = parse_args()
    imgs_dir = params.input_dir
    avatars_dir = params.avatars_dir
    output_dir = params.output_dir

    os.makedirs(output_dir, exist_ok=True)

    example_images = [f for f in os.listdir(imgs_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    avatar_images = [f for f in os.listdir(avatars_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--window-size=375,800')
    options.add_argument('--allow-file-access-from-files')
    driver = webdriver.Chrome(options=options)


    for i, example in enumerate(example_images):
        example_path = os.path.join(imgs_dir, example)
        avatar = random.choice(avatar_images)
        avatar_path = os.path.join(avatars_dir, avatar)
        filename, file_extension = os.path.splitext(example)
        output_filename = f"{filename}{file_extension}"
        output_path = os.path.join(output_dir, output_filename)
        generate_post(driver, example_path, avatar_path, output_path)
    
    driver.quit()
