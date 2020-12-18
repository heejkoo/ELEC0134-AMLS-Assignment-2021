import os
import cv2
import random
import numpy as np
from natsort import natsorted

from sklearn.model_selection import train_test_split

from imutils.face_utils import rect_to_bb
from dlib import get_frontal_face_detector, shape_predictor

import tensorflow as tf


def set_global_determinism(seed, fast_n_close=False):
    """For Reproducible Tensorflow"""

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    if fast_n_close:
        return

    from tfdeterminism import patch

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    patch()


def load_images(path_name):
    """Load images in a numpy format"""

    # Image Path
    image_path = os.path.join("../Datasets/", path_name + "/img/")
    image_paths = list()

    # Add images to list
    for i in os.listdir(image_path):
        image_paths.append(os.path.join(image_path, i))

    # Sort images
    image_paths = natsorted(image_paths)

    # List to add images in a numpy format
    images = list()

    # Preprocess
    for image_path in image_paths:
        # Preprocess images, cropping to a specific size
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        # Preprocess images in a numpy format
        image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.float32)
        # Normalize (0 to 1)
        image = image / 255.
        images.append(image)

    images = np.array(images)

    return images


def load_labels(path_name, classification):
    """Load labels in a numpy format"""

    # Labels Path
    label_path = os.path.join("../Datasets/", path_name + "/labels.csv")
    label_file = open(label_path, "r")

    # Read line by line
    label_lines = label_file.readlines()
    label_file.close()

    # Split by space (blank)
    sort = label_lines[0].split().index(classification)

    # List to add labels in a numpy format
    labels = list()

    # Preprocess
    for label_line in label_lines[1:]:
        label_line = label_line.split()

        # Convert -1 and 1 to 0 and 1
        if classification == 'gender':
            label = (int(label_line[sort + 1]) + 1) / 2
        elif classification == 'smiling':
            label = (int(label_line[sort + 1]) + 1) / 2
        elif classification == 'face_shape':
            label = int(label_line[sort + 1])
        elif classification == 'eye_color':
            label = int(label_line[sort + 1])

        labels.append(label)

    labels = np.array(labels, dtype=np.int8)

    return labels


def facial2np(shape):
    landmarks = np.zeros((shape.num_parts, 2), dtype="int")
    for i in range(0, shape.num_parts):
        landmarks[i] = (shape.part(i).x, shape.part(i).y)
    return landmarks


def extract_landmarks(image, face_detector, shape_pred):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("uint8")

    bboxes = face_detector(image, 1)
    num_faces = len(bboxes)

    if num_faces == 0:
        return None

    num_landmarks = 68

    areas = np.zeros((1, num_faces))
    shapes = np.zeros((num_landmarks * 2, num_faces), dtype=np.int64)

    for i, bbox in enumerate(bboxes):
        pred = shape_pred(image, bbox)
        pred = facial2np(pred)

        (_, _, w, h) = rect_to_bb(bbox)
        shapes[:, i] = np.reshape(pred, [num_landmarks * 2])
        areas[0, i] = w * h

    out = np.reshape(np.transpose(shapes[:, np.argmax(areas)]), [num_landmarks, 2])
    return out


def get_landmarks(images, labels):
    landmarks_detected, labels_detected = list(), list()

    face_detector = get_frontal_face_detector()
    shape_pred = shape_predictor('../Datasets/shape_predictor_68_face_landmarks.dat')

    for i, image in enumerate(images):
        out = extract_landmarks(image * 255.0, face_detector, shape_pred)
        if out is None:
            pass
        else:
            landmarks_detected.append(out)
            labels_detected.append(labels[i])

    landmarks_detected = np.array(landmarks_detected, dtype=np.int16)
    labels_detected = np.array(labels_detected, dtype=np.int16)

    return landmarks_detected, labels_detected


def check_imbalance(path_name, classification, num_classes):
    """Check Imbalance"""

    # Labels Path
    label_path = os.path.join("../Datasets/", path_name, "labels.csv")
    label_file = open(label_path, "r")

    # Read line by line
    label_lines = label_file.readlines()
    label_file.close()

    # Split by space (blank)
    sort = label_lines[0].split().index(classification)

    # If number of classes is 2,
    if num_classes == 2:
        class_1, class_2 = list(), list()

        for label_line in label_lines[1:]:
            label_line = label_line.split()

            label = int(label_line[sort + 1])

            if label == -1:
                class_1.append(label)
            elif label == 1:
                class_2.append(label)

        print("The number of each class is {} and {}.".
              format(len(class_1), len(class_2)))

    # If number of classes is 5,
    elif num_classes == 5:
        class_0, class_1, class_2, class_3, class_4 = list(), list(), list(), list(), list()

        for label_line in label_lines[1:]:
            label_line = label_line.split()

            label = int(label_line[sort + 1])

            if label == 0:
                class_0.append(label)
            elif label == 1:
                class_1.append(label)
            elif label == 2:
                class_2.append(label)
            elif label == 3:
                class_3.append(label)
            elif label == 4:
                class_4.append(label)

        print("The number of each class is {}, {}, {}, {} and {}.".
              format(len(class_0), len(class_1), len(class_2), len(class_3), len(class_4)))

    else:
        print("The number of classes can be 2 or 5 in this assignment.")


def data_preprocessing(datasets, classification, train_size=0.8, test_size=0.5, random_state=42, landmarks=False):
    """Data Pre-processing"""

    # Load images and labels
    images = load_images(datasets)
    labels = load_labels(datasets, classification)

    if landmarks:
        images, labels = get_landmarks(images, labels)

    # Split the data set to train set and validation set
    train_images, val_images, train_labels, val_labels = train_test_split(images,
                                                                          labels,
                                                                          train_size=train_size,
                                                                          stratify=labels,
                                                                          random_state=random_state)

    # Split the train set to validation set and test set
    val_images, test_images, val_labels, test_labels = train_test_split(val_images,
                                                                        val_labels,
                                                                        test_size=test_size,
                                                                        stratify=val_labels,
                                                                        random_state=random_state)

    # Crop images if necessary
    if classification == 'face_shape':
        train_images = tf.image.crop_to_bounding_box(train_images, offset_height=28, offset_width=28, target_height=72, target_width=72)
        val_images = tf.image.crop_to_bounding_box(val_images, offset_height=28, offset_width=28, target_height=72, target_width=72)
        test_images = tf.image.crop_to_bounding_box(test_images, offset_height=28, offset_width=28, target_height=72, target_width=72)
    elif classification == 'eye_color':
        train_images = tf.image.crop_to_bounding_box(train_images, offset_height=60, offset_width=40, target_height=18, target_width=50)
        val_images = tf.image.crop_to_bounding_box(val_images, offset_height=60, offset_width=40, target_height=18, target_width=50)
        test_images = tf.image.crop_to_bounding_box(test_images, offset_height=60, offset_width=40, target_height=18, target_width=50)

    # Print out the number of data in each data set
    print("The number of train images :", len(train_images))
    print("The number of validation images: ", len(val_images))
    print("The number of test images: ", len(test_images))

    # Wrap images and labels
    train_set = (train_images, train_labels)
    val_set = (val_images, val_labels)
    test_set = (test_images, test_labels)

    return train_set, val_set, test_set