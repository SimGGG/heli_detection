import os
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', '-i', dest='image_dir', default=None)
parser.add_argument('--class_name', '-c', dest='class_name', default=None)

args = parser.parse_args()


'''
Examples

python image_rename.py \
-i /home/user/Helipad_Detection/workspace/images/vehicle \
-c vehicle

'''

IMAGE_DIR = args.image_dir
CLASS_NAME = args.class_name


src_img_list = sorted([x for x in os.listdir(IMAGE_DIR) if '.xml' not in x])

def mod_xml(dst_xml, folder, filename, path):
    tree = ET.parse(dst_xml)
    root = tree.getroot()

    root.find('folder').text = folder
    root.find('filename').text = filename
    root.find('path').text = path

    tree.write(dst_xml)

for idx, img in enumerate(src_img_list):

    try:
        fnm, ext = img.split('.')

        src_img = os.path.join(IMAGE_DIR, img)
        dst_img = os.path.join(IMAGE_DIR, f"{CLASS_NAME}{idx}.{ext}")

        src_xml = os.path.join(IMAGE_DIR, f"{fnm}.xml")
        dst_xml = os.path.join(IMAGE_DIR, f"{CLASS_NAME}{idx}.xml")

        os.rename(src_img, dst_img)
        os.rename(src_xml, dst_xml)

        # modifying .xml

        folder = IMAGE_DIR.split('/')[-1]
        mod_xml(dst_xml, folder, f"{CLASS_NAME}{idx}.{ext}", dst_img)
    except:
        print(f'Error in {src_img} / {dst_img}')







