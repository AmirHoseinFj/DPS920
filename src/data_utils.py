"""Loading, balancing, and batching of the driving_log.csv dataset."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from augmentation import augment_image
from preprocessing import preprocess_image

COLUMNS = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']


def load_log(csv_path):
    data = pd.read_csv(csv_path, names=COLUMNS)
    for cam in ('center', 'left', 'right'):
        data[cam] = data[cam].apply(lambda p: os.path.basename(str(p).strip()))
    return data


def expand_cameras(data, correction=0.2):
    """left/right cameras into the dataset as extra samples

    This is the recovery trick from the Nvidia paper. The side cameras see the
    road from a position offset from the car's centre, so if we label a left
    image with (steering + correction) and a right image with
    (steering - correction), the model learns to steer back to the middle
    whenever it drifts, without having to record recovery driving by
    hand. It also triples the amount of training data.
    """
    frames = [
        pd.DataFrame({'image': data['center'], 'steering': data['steering']}),
        pd.DataFrame({'image': data['left'],
                      'steering': data['steering'] + correction}),
        pd.DataFrame({'image': data['right'],
                      'steering': data['steering'] - correction}),
    ]
    out = pd.concat(frames, ignore_index=True)
    out['steering'] = out['steering'].clip(-1.0, 1.0)
    return out


def balance_data(data, bins=25, max_per_bin=200, show_plot=False):
    hist, edges = np.histogram(data['steering'], bins)

    keep_indices = []
    for i in range(bins):
        bin_indices = data[(data['steering'] >= edges[i]) &
                            (data['steering'] <= edges[i + 1])].index.tolist()
        np.random.shuffle(bin_indices)
        keep_indices.extend(bin_indices[:max_per_bin])

    balanced = data.loc[keep_indices].reset_index(drop=True)

    if show_plot:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(edges[:-1], hist, width=edges[1] - edges[0])
        axes[0].set_title('Before balancing')
        hist2, _ = np.histogram(balanced['steering'], bins)
        axes[1].bar(edges[:-1], hist2, width=edges[1] - edges[0])
        axes[1].set_title('After balancing')
        plt.tight_layout()
        plt.savefig('steering_histogram.png')
        plt.close()

    return balanced


def train_valid_split(data, img_dir, valid_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    split = int(len(data) * (1 - valid_frac))
    train_idx, valid_idx = indices[:split], indices[split:]

    col = 'image' if 'image' in data.columns else 'center'
    image_paths = np.array([os.path.join(img_dir, f) for f in data[col]])
    steerings = np.array(data['steering'], dtype=np.float32)

    return (image_paths[train_idx], steerings[train_idx],
            image_paths[valid_idx], steerings[valid_idx])


def batch_generator(image_paths, steerings, batch_size, is_training):
    while True:
        batch_imgs, batch_steer = [], []
        indices = np.random.randint(0, len(image_paths), batch_size)
        for i in indices:
            img = mpimg.imread(image_paths[i])
            steer = steerings[i]
            if is_training:
                img, steer = augment_image(img, steer)
            img = preprocess_image(img)
            batch_imgs.append(img)
            batch_steer.append(steer)
        yield np.asarray(batch_imgs), np.asarray(batch_steer)
