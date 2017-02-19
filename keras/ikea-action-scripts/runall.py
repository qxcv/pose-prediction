#!/usr/bin/env python3

"""Run a VGG-like action classifier network over the entire IkeaDB dataset,
saving features to a specified directory."""

from multiprocessing import Pool
from os import walk, path
from h5py import File
from scipy.misc import imread, imresize
import numpy as np
from keras.models import load_model

MODEL_PATH = './ikea-action-network.h5'
SRC = '/data/home/cherian/IkeaDataset/Frames/'
DEST = './vectors.h5'
BATCH_SIZE = 64


def get_paths():
    for dirpath, dirs, filenames in walk(SRC, followlinks=True):
        if '256' in dirs:
            # don't visit directories for resized files
            dirs.remove('256')
        for fn in filenames:
            ext = '.jpg'
            if fn.endswith(ext):
                src_path = path.join(SRC, dirpath, fn)
                if dirpath.startswith(SRC):
                    reldir = dirpath[len(SRC):]
                else:
                    reldir = dirpath
                dest_ident = path.join(reldir, fn[:-len(ext)])
                # dirpath will be used for storage, full_path will be used to read the image
                yield src_path, dest_ident


def resize_image(args):
    src_path, dest_ident = args

    loaded_image = imread(src_path)
    h, w, c = loaded_image.shape

    # My cropping operations are hacky. Still, this is much easier than doing
    # it "the right way" (i.e. using actual person boxes)
    if h > w:
        # Portrait: crop bottoms to make it square
        total_loss = h - w
        top_loss = total_loss // 2
        bottom_loss = total_loss - top_loss
        cropped = loaded_image[top_loss:h-bottom_loss]
    else:
        # Landscape: crop left and right to make it square
        total_loss = w - h
        left_loss = total_loss // 2
        right_loss = total_loss - left_loss
        cropped = loaded_image[:, left_loss:w-right_loss]

    assert cropped.shape[0] == cropped.shape[1], "Image should be square after cropping"

    # Resize to VGG input shape and use Caffe dimension ordering
    resized = imresize(cropped, (224, 224))
    chan_first = resized.transpose((2, 0, 1))
    assert chan_first.shape == (3, 224, 224), chan_first.shape

    return chan_first, dest_ident


def main():
    print('Loading model')
    model = load_model(MODEL_PATH)

    print('Running read loop')
    with File(DEST) as fp, Pool() as p:
        iterator = p.imap_unordered(resize_image, get_paths())
        # Recall that we're using Caffe ordering. Don't think it's BGR (I hope
        # not, anyway!).
        batch = np.zeros((BATCH_SIZE, 3, 224, 224))
        dest_idents = [None] * BATCH_SIZE
        buffer_count = 0
        keep_going = True

        while keep_going:
            process_buffer = False

            try:
                batch[buffer_count], dest_idents[buffer_count] = next(iterator)
                if dest_idents[buffer_count] not in fp:
                    # we only bother increasing buffer_count when we see
                    # something not in the destination file
                    buffer_count += 1
                    process_buffer = process_buffer or buffer_count >= BATCH_SIZE
                else:
                    print('Ignoring %s' % dest_idents[buffer_count])
            except StopIteration:
                process_buffer = True
                keep_going = False

            # Time to run queued images through model
            if process_buffer:
                print('Running batch predictions (%d of them)' % buffer_count)
                out_vectors = model.predict(batch).astype('float32')

                print('Saving batch predictions')
                zipper = zip(out_vectors, dest_idents)
                print(', '.join(map(str, dest_idents)))
                for vector, where_to_store in zipper:
                    if where_to_store != None:
                        fp[where_to_store] = vector

                print('Refreshing batch storage')
                batch = np.zeros((BATCH_SIZE, 3, 224, 224))
                dest_idents = [None] * BATCH_SIZE
                buffer_count = 0

        print('Finished')


if __name__ == '__main__':
    main()
