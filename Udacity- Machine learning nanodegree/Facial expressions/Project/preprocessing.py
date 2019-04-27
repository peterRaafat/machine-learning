import random
from typing import List

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
        r = random.randint(0, size-1)
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
    cropped_labels = np.empty([0, 7])
    cropped_images = []
    for i, img in enumerate(images):
        if i % 10 == 0:
            print('\r{}/{}'.format(i, len(images)), end="", flush=True)
        faces = face_detector(img)
        if(len(faces) == 1) and (faces[0][2] > 50) and (faces[0][3] > 50):
            # crop the image
            x, y, w, h = faces[0]
            array_image = np.array(img)
            cropped_images.append(array_image[y:y + h, x:x + w])
            cropped_labels = np.append(cropped_labels, [labels[i]], axis = 0)
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

def JAFFE_count(raw_labels):
    num_neutral = raw_labels.count('NE')
    num_happy = raw_labels.count('HA')
    num_sad = raw_labels.count('SA')
    num_surprised = raw_labels.count('SU')
    num_angry = raw_labels.count('AN')
    num_disgust = raw_labels.count('DI')
    num_fear = raw_labels.count('FE')
    return num_neutral, num_happy, num_sad, num_surprised, num_angry, num_disgust, num_fear



def cohn_count(labels):
    num_angry = num_happy = num_sad = num_surprised = num_disgust = num_fear = num_contempt = 0

    for label in labels:
        label = np.array(label)
        index = np.where(label == 1)[0][0]
        if index == 0:
            num_angry += 1
        if index == 1:
            num_disgust += 1
        if index == 2:
            num_fear += 1
        if index == 3:
            num_happy +=1
        if index == 4:
            num_sad += 1
        if index == 5:
            num_surprised += 1
        if index == 6:
            num_contempt +=1
    return num_angry, num_happy, num_sad, num_surprised, num_disgust, num_fear, num_contempt

def cohn_augment(images, labels, desired_size):
    size = len(images)
    augmented_images = images
    augmented_labels = labels
    for generated_number in range(desired_size - size):
        print('\r{}/{}'.format(generated_number, desired_size - size), end="", flush=True)
        # pick a random image from raw images
        r = random.randint(0, size-1)
        im = images[r]
        # add the label ot augmenetd labels
        augmented_labels = np.append(augmented_labels, [labels[r]], axis=0)
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

def jaffe_reorder(jaffe_labels):
    ret = []  # type: List[List[int]]
    for label in jaffe_labels:
        if np.argmax(label) == 1:
            ret.append([0, 0, 0, 1, 0, 0, 0])
        if np.argmax(label) == 2:
            ret.append([0, 0, 0, 0, 1, 0, 0])
        if np.argmax(label) == 3:
            ret.append([0, 0, 0, 0, 0, 1, 0])
        if np.argmax(label) == 4:
            ret.append([1, 0, 0, 0, 0, 0, 0])
        if np.argmax(label) == 5:
            ret.append([0, 1, 0, 0, 0, 0, 0])
        if np.argmax(label) == 6:
            ret.append([0, 0, 1, 0, 0, 0, 0])
    return ret

def remove_neutral(jaffe_images, jaffe_labels):
    jaffe_images = [np.array(img) for img in jaffe_images]
    ret_images = []
    ret_labels = []
    indicies = []
    for i, label in enumerate(jaffe_labels):
        if np.argmax(label) == 0:
            indicies.append(i)
    ret_images = np.delete(jaffe_images, indicies, axis = 0)
    ret_labels = np.delete(jaffe_labels, indicies, axis = 0)
    return ret_images, ret_labels

def cohn_rearrange(labels):
    ret = labels
    for label in ret:
        label[0], label[6] = label[6], label[0]
    return ret
