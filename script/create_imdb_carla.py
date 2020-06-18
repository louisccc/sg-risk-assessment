import argparse, os, json, string
from queue import Queue
from threading import Thread, Lock

import h5py
import numpy as np
from cv2 import imread
from PIL import Image
from pathlib import Path

def build_filename_dict(data):
    # First make sure all basenames are unique
    basenames_list = [os.path.basename(img['image_path']) for img in data]
    assert len(basenames_list) == len(set(basenames_list))

    next_idx = 1
    filename_to_idx, idx_to_filename = {}, {}
    for img in data:
        filename = os.path.basename(img['image_path'])
        filename_to_idx[filename] = next_idx
        idx_to_filename[next_idx] = filename
        next_idx += 1
    return filename_to_idx, idx_to_filename


def encode_filenames(data, filename_to_idx):
    filename_idxs = []
    for img in data:
        filename = os.path.basename(img['image_path'])
        idx = filename_to_idx[filename]
        filename_idxs.append(idx)
    return np.asarray(filename_idxs, dtype=np.int32)


def add_images(h5_file, args):
    fns = []; ids = []; idx = []; lcids = []

    img_dir = Path(args.image_dir).resolve()
    for i, filepath in enumerate(img_dir.glob('**/raw_images/*'+args.img_format)):
        fns.append(str(filepath))
        lcids.append(int(filepath.parts[-3]))
        ids.append(int(filepath.stem))
        idx.append(i)

    ids = np.array(ids, dtype=np.int32)
    idx = np.array(idx, dtype=np.int32)
    lcids = np.array(lcids, dtype=np.int32)

    h5_file.create_dataset('image_ids', data=ids)
    h5_file.create_dataset('valid_idx', data=idx)
    h5_file.create_dataset('lanechange_ids', data=lcids)

    num_images = len(fns)

    shape = (num_images, 3, args.image_size, args.image_size)
    image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
    original_heights = np.zeros(num_images, dtype=np.int32)
    original_widths = np.zeros(num_images, dtype=np.int32)
    image_heights = np.zeros(num_images, dtype=np.int32)
    image_widths = np.zeros(num_images, dtype=np.int32)

    lock = Lock()
    q = Queue()
    for i, fn in enumerate(fns):
        q.put((i, fn))

    def worker():
        while True:
            i, filename = q.get()

            if i % 10000 == 0:
                print('processing %i images...' % i)

            img = imread(filename)
            H0, W0, _ = img.shape

            img = np.array(Image.fromarray(img).resize((args.image_size, args.image_size)))
            H, W, _ = img.shape

            lock.acquire()
            original_heights[i] = H0
            original_widths[i] = W0
            image_heights[i] = H
            image_widths[i] = W
            image_dset[i, :, :H, :W] = img.transpose(2, 0, 1)
            lock.release()
            q.task_done()

    for i in range(args.num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    q.join()

    h5_file.create_dataset('image_heights', data=image_heights)
    h5_file.create_dataset('image_widths', data=image_widths)
    h5_file.create_dataset('original_heights', data=original_heights)
    h5_file.create_dataset('original_widths', data=original_widths)

    return fns


def main(args):
    h5_fn = 'imdb_carla_' + str(args.image_size) + '.h5'
    # write the h5 file
    h5_file = os.path.join(args.imh5_dir, h5_fn)
    f = h5py.File(h5_file, 'w')
    # load images
    im_fns = add_images(f, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='/home/aung/NAS/louisccc/av/synthesis_data/lane-change-100-old/')
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--imh5_dir', default='.')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--img_format', default='.png', type=str)

    args = parser.parse_args()
    main(args)