from os import listdir
import argparse
import tqdm

import numpy as np
import xml.etree.ElementTree as ET

import cv2
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
from files import *
from pascal_voc_writer import Writer


def read_annotation(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bounding_box_list = []

    file_name = root.find('filename').text
    for obj in root.iter('object'):

        object_label = obj.find("name").text
        for box in obj.findall("bndbox"):
            x_min = int(box.find("xmin").text)
            y_min = int(box.find("ymin").text)
            x_max = int(box.find("xmax").text)
            y_max = int(box.find("ymax").text)

        bounding_box = [object_label, x_min, y_min, x_max, y_max]
        bounding_box_list.append(bounding_box)

    return bounding_box_list, file_name

def read_train_dataset(dir):
    images = []
    annotations = []

    for file in tqdm.tqdm(listdir(dir)):
        if 'jpg' in file.lower() or 'png' in file.lower() or 'jpeg' in file.lower():
            images.append(cv2.imread(dir + file, 1))
            annotation_file = file.replace(file.split('.')[-1], 'xml')
            bounding_box_list, file_name = read_annotation(dir + annotation_file)
            annotations.append((bounding_box_list, annotation_file, file_name))

    images = np.array(images)
    return images, annotations


def read_bbs(image, annotation):
    boxes = annotation[0]
    ia_bounding_boxes = []
    for box in boxes:
        ia_bounding_boxes.append(ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4]))
    bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)
    return bbs


def do_augmentation(image, bbs):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order


    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    return image_aug, bbs_aug


def check_augmentation_images(image, annotation):

    bbs = read_bbs(image, annotation)
    image_aug, bbs_aug = do_augmentation(image, bbs)

    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
              )

    image_before = bbs.draw_on_image(image, thickness=1)
    image_after = bbs_aug.draw_on_image(image_aug, thickness=1, color=[0, 0, 255])

    cv2.imshow('image_before', cv2.resize(image_before, (380, 640)))
    cv2.imshow('image_after', cv2.resize(image_after, (380, 640)))
    cv2.waitKey(0)


def save_augmentation_images(images, annotations, dir):
    for idx in tqdm.tqdm(range(len(images))):
        bbs = read_bbs(images[idx], annotations[idx])
        file_idx = 0
        for _ in range(30):
            image_aug, bbs_aug = do_augmentation(images[idx], bbs)
            new_image_file = dir + f'aug_{file_idx}_' + annotations[idx][2]
            cv2.imwrite(new_image_file, image_aug)

            h, w = np.shape(image_aug)[0:2]
            voc_writer = Writer(new_image_file, w, h)

            for i in range(len(bbs_aug.bounding_boxes)):
                bb_box = bbs_aug.bounding_boxes[i]
                voc_writer.addObject(annotations[idx][0][i][0], int(bb_box.x1), int(bb_box.y1), int(bb_box.x2), int(bb_box.y2))

            voc_writer.save(dir + f'aug_{file_idx}_' + annotations[idx][1])

            file_idx += 1


def main():
    parser = argparse.ArgumentParser(
        description="Augmentation with Labeled Images")
    parser.add_argument("-i",
                        "--image_dir",
                        help="Path to the folder where the input image files are stored. "
                             "Defaults to the same directory as XML_DIR.",
                        type=str, default=None)

    args = parser.parse_args()

    print('Read Datasets.. ')
    images, annotations = read_train_dataset(args.image_dir)


    # check_augmentation_images(images[3], annotations[3])
    print('\nDo Augmentation.. ')
    save_augmentation_images(images, annotations, args.image_dir)



if __name__ == "__main__":
    main()
