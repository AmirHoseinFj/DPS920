"""Random data augmentation applied to training images only.

Each transform is applied independently with its own probability so a
given image receives a mix of a few transforms, not all of
them, this keeps augmented data varied instead of uniformly distorted.
"""
import cv2
import numpy as np


FLIP_PROB = 0.5

# Steering units to add per fraction-of-image-width of horizontal shift.
# A pan that slides the road to the right makes the car looks as if it sits
# further left, so the correct label steer back to the right.
PAN_STEER_PER_WIDTH = 0.4

# steering units per degree of rotation
ROTATE_STEER_PER_DEG = 0.006


def pan(img, steering):
    shift_frac = np.random.uniform(-0.1, 0.1)
    tx = shift_frac * img.shape[1]
    ty = np.random.uniform(-0.1, 0.1) * img.shape[0]
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    out = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
    return out, steering - shift_frac * PAN_STEER_PER_WIDTH


def zoom(img):
    # Zoom is centred, so it does not move the road sideways
    scale = np.random.uniform(1.0, 1.3)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(img, m, (w, h))


def rotate(img, steering, max_angle=5):
    #rotate a little, correcting steering for the heading change
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h))
    return out, steering - angle * ROTATE_STEER_PER_DEG


def brightness(img):
    factor = np.random.uniform(0.4, 1.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def flip(img, steering):
    return cv2.flip(img, 1), -steering


def augment_image(img, steering, prob=0.5):
    if np.random.rand() < prob:
        img, steering = pan(img, steering)
    if np.random.rand() < prob:
        img = zoom(img)
    if np.random.rand() < prob:
        img = brightness(img)
    if np.random.rand() < prob:
        img, steering = rotate(img, steering)
    if np.random.rand() < FLIP_PROB:
        img, steering = flip(img, steering)
    return img, float(np.clip(steering, -1.0, 1.0))
