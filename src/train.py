"""
Train the Nvidia CNN on collected driving_log.csv data and save model.h5.

"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorflow.keras.callbacks import ModelCheckpoint

from data_utils import (balance_data, batch_generator, expand_cameras, load_log,
                        train_valid_split)
from model import build_nvidia_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='driving_log.csv')
    p.add_argument('--img-dir', default='IMG')
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--steps-per-epoch', type=int, default=300)
    p.add_argument('--epochs', type=int, default=15)
    # 400 rather than 200: with all three cameras folded in there are ~3x as
    # many samples, so the old cap threw away most of the usable data
    p.add_argument('--max-per-bin', type=int, default=400)
    p.add_argument('--correction', type=float, default=0.2,
                   help='steering offset applied to the left/right cameras')
    p.add_argument('--center-only', action='store_true',
                   help='ignore the left/right cameras (worse recovery)')
    p.add_argument('--out', default='model.h5')
    return p.parse_args()


def main():
    args = parse_args()

    print('Loading driving log...')
    data = load_log(args.csv)
    print(f'{len(data)} rows loaded')

    if not args.center_only:
        data = expand_cameras(data, correction=args.correction)
        print(f'{len(data)} samples after folding in left/right cameras '
              f'(correction={args.correction})')

    data = balance_data(data, max_per_bin=args.max_per_bin, show_plot=True)
    print(f'{len(data)} rows after balancing (histogram saved to steering_histogram.png)')

    x_train, y_train, x_valid, y_valid = train_valid_split(data, args.img_dir)
    print(f'Train: {len(x_train)}  Valid: {len(x_valid)}')

    model = build_nvidia_model()
    model.summary()

    train_gen = batch_generator(x_train, y_train, args.batch_size, is_training=True)
    valid_gen = batch_generator(x_valid, y_valid, args.batch_size, is_training=False)

    validation_steps = max(1, len(x_valid) // args.batch_size)

    # Validation loss bounces from epoch to epoch because each batch is a fresh
    # random draw with random augmentation. Keeping only the best-scoring epoch
    # avoids shipping whatever the last epoch happened to be, which can be
    # noticeably worse than the best checkpoint seen during the run. 
    # I added it because of experiencing this issue.
    checkpoint = ModelCheckpoint(args.out, monitor='val_loss', mode='min',
                                 save_best_only=True, verbose=1)

    history = model.fit(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        callbacks=[checkpoint],
    )

    best = min(history.history['val_loss'])
    print(f'Best model (val_loss {best:.4f}) saved to {args.out}')

    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('training_history.png')
    print('Training curves saved to training_history.png')


if __name__ == '__main__':
    main()
