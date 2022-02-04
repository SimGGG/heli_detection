import io
import os
import glob
import scipy.misc
import numpy as np
import six
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
from object_detection.utils import visualization_utils as viz_utils

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-m', dest='model_path', default=None)
parser.add_argument('--image_dir', '-i', dest='image_dir', default=None)

args = parser.parse_args()
model_path = os.path.join(args.model_path, 'saved_model/')
image_dir = args.image_dir

image_path = []
for ext in ('*.png', '*.jpg', '*.jpeg'):
    image_path += glob.glob(os.path.join(image_dir, ext))

category_index = {1 : {'id' : 1, 'name':'helipad'}}

# model_path = './exported_models/heli_ssd_mobile_v2_0104/saved_model'
# test_image_dir = './images/test'
# test_image_path = glob.glob(os.path.join(test_image_dir, '*.png'))
# category_index = {1 : {'id' : 1, 'name':'helipad'}}


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path (this can be local or on colossus)

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



# Load model
def load_model(model_path):
    start_time = time.time()
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(model_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time: ' + str(elapsed_time) + 's')
    return detect_fn


# Inference
def inference_model(detect_fn, image):
    image_np = load_image_into_numpy_array(image)
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()
    # elapsed.append(end_time - start_time)

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.40,
            agnostic_mode=False)

    # cv2.imshow('image_before', cv2.resize(image_np, (380, 640)))
    # cv2.imshow('image_after', cv2.resize(image_np, (380, 640)))
    # cv2.waitKey(0)
    P_image = Image.fromarray(image_np_with_detections)
    P_image.show()



if __name__ == "__main__":

    detect_fn = load_model(model_path)
    elapsed = []
    rand_idx = np.random.choice(len(image_path), 1)
    image_path = image_path[rand_idx[0]]
    inference_model(detect_fn, image_path)

    # if len(image_path) >= 5:
    #     random_indices = np.random.choice(len(image_path), 5, replace=False)
    #     image_path = [image_path[idx] for idx in random_indices]
    #
    # for image in image_path:
    #     inference_model(detect_fn, image)

    # mean_elapsed = sum(elapsed) / float(len(elapsed))
    # print('Elapsed time: ' + str(mean_elapsed) + ' second per image')
