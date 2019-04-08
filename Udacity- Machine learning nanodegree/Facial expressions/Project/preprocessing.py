import random
import skimage as sk
from skimage import transform as tf
from skimage import img_as_ubyte
from skimage import util
import numpy as np
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


def random_noise(image):
    image = np.array(image)
    im = sk.util.random_noise(image)
    im = img_as_ubyte(im)
    return im


def random_rotation(image):
    image = np.array(image)
    random_degree = random.uniform(-15, 15)
    im = (sk.transform.rotate(image, random_degree))
    im = img_as_ubyte(im)
    return im


def random_skew(image):
    shear = random.uniform(-0.2, 0.2)
    afine_tf = tf.AffineTransform(shear=shear)
    im = tf.warp(image, inverse_map=afine_tf)
    im = img_as_ubyte(im)
    return im


def augment(images, labels, desired_size):
    size = len(images)
    augmented_images = images
    augmented_labels = labels
    for generated_number in range(desired_size - size):
        # pick a random image from raw images
        r = random.randint(0, size)
        im = images[r]
        # add the label ot augmenetd labels
        augmented_labels.append(labels[r])
        # pick a random trasnformation
        trans = random.randint(1, 3)
        if trans == 1:
            # apply random gaussian noise
            augmented_images.append(random_noise(im))
        elif trans == 2:
            # apply random rotation
            augmented_images.append(random_rotation(im))
        else:
            # apply random skew
            augmented_images.append(random_skew(im))
    return augmented_images, augmented_labels


def JAFFE_encode(labels):
    encoded_labels = np.zeros((len(labels), 7))
    for i, value in enumerate(labels):
        if value == 'NE':
            encoded_labels[i][0] = 1
        elif value == 'HA':
            encoded_labels[i][1] = 1
        elif value == 'SA':
            encoded_labels[i][2] = 1
        elif value == 'SU':
            encoded_labels[i][3] = 1
        elif value == 'AN':
            encoded_labels[i][4] = 1
        elif value == 'DI':
            encoded_labels[i][5] = 1
        elif value == 'FE':
            encoded_labels[i][6] = 1
        else:
            print('Invalid Label')
    return encoded_labels


def face_detector(img):
    img = np.array(img)
    return face_cascade.detectMultiScale(img)


def crop_face(images, labels):
    cropped_labels = labels
    cropped_images = []
    for i, img in enumerate(images):
        if i % 10 == 0:
            print('\r{}/{}'.format(i, len(images)), end="", flush=True)
        faces = face_detector(img)
        if len(faces) != 1:
            cropped_labels = np.delete(cropped_labels, i, axis=0)
        elif faces[0][2] < 50 or faces[0][3] < 50:
            cropped_labels = np.delete(cropped_labels, i, axis=0)
        else:
            # crop the image
            x, y, w, h = faces[0]
            array_image = np.array(img)
            cropped_images.append(array_image[y:y + h, x:x + w])
    return cropped_images, cropped_labels


def resize(images, size):
    return [Image.fromarray(images[i]).resize((size, size), Image.BILINEAR) \
            for i in range(len(images))]


def normalize(img):
    img = np.array(img)
    img_min = np.min(img)
    img_max = np.max(img)
    img = (img - img_min) / float(img_max - img_min)
    return img
