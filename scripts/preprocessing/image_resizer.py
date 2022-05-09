import os
import re
import argparse
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', '-i', dest='image_dir', default=None)
args = parser.parse_args()

IMAGE_DIR = args.image_dir
OUTPUT_IMAGE_DIR = IMAGE_DIR + "_resized"

if not os.path.exists(OUTPUT_IMAGE_DIR):
    os.mkdir(OUTPUT_IMAGE_DIR)

IMAGES = [f for f in os.listdir(IMAGE_DIR) if re.search(r'(.jpg|.png|.jpeg|.JPG|.PNG|.JPEG)', f)]


for img in IMAGES:
    try:
        img_path = os.path.join(IMAGE_DIR, img)
        image = Image.open(img_path)
        resized_image = image.resize((1024, 1024))
        resized_image.save(os.path.join(OUTPUT_IMAGE_DIR, img))
    except:
        print(img)



# img_path = os.path.join('/home/user/Helipad_Detection/workspace/images/', 'helipad_1.jpg')
# image = Image.open(img_path)
# resized_image = image.resize((1024, 1024))
# resized_image.save(os.path.join('/home/user/Helipad_Detection/workspace/images/', 'helipad_1_out.jpg'))