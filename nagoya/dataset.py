import pickle
import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
# FIXME: Not working... ModuleNotFoundError: No module named 'prompt_toolkit.formatted_text'
# from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects
from pathlib import Path
from collections import Counter

class DataSet:

    def __init__(self):
        # TODO: clean these up, some are not used
        # Better to just refactor the entire thing...
        self.dataset = {}
        self.video = []
        self.foldernames = []
        self.image_seq = []
        self.risk_scores = []
        self.risk_one_hot = []
        self.risk_binary = []

    @classmethod
    def loader(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def preprocess_clips(self, data_dir):
        foldernames = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir / f)]
        foldernames = [f for f in foldernames if f.split('_')[0].isnumeric()]
        foldernames = sorted(foldernames, key=self.get_clipnumber)
        # only grab dataset clips (ignore filtered clips)
        self.foldernames = self.get_filtered_clips(data_dir, foldernames)
        # for visualizing amount of frames in each clip of the dataset
        self.get_max_frames(data_dir)

    def read_video(self, data_dir, option='fixed frame amount', number_of_frames=20, max_number_of_frames=500,
                   scaling='no scaling', scale_x=0.1, scale_y=0.1):
        
        self.preprocess_clips(data_dir)
        is_valid = self.valid_dataset(data_dir/self.foldernames[0], scaling, scale_x, scale_y)
        if is_valid:
            # shape: (n_videos, n_frames, im_height, im_width, channel)
            im_height, im_width, channel = self.image_seq[0].shape
            if option == 'fixed frame amount':
                self.video = np.zeros([len(self.foldernames), number_of_frames, im_height, im_width, channel])
            elif option == 'all frames':
                self.video = np.zeros([len(self.foldernames), max_number_of_frames, im_height, im_width, channel])

            # todo convert this to a wrapper
            for idx, foldername in tqdm(enumerate(self.foldernames)):
                if foldername.isnumeric:
                    is_valid = self.valid_dataset(str(data_dir/foldername), scaling=scaling, scale_x=scale_x, scale_y=scale_y)
                    print(foldername)
                    if is_valid:
                        if option == 'fixed frame amount':
                            self.video[idx, :, :, :, :] = self._read_video_helper(number_of_frames=number_of_frames)
                        elif option == 'all frames':
                            self.video[idx, 0:len(self.image_seq), :, :, :] = self.image_seq
        else:
            raise Exception('Error reading first clip! Check path or contents of {}'.format(data_dir))
    
    def _read_video_helper(self, number_of_frames=20):
        images = []
        index = 0
        # length of image sequence must be greater than or equal to number_of_frames
        # if number of frames is less than entire length of image sequence, takes every nth frame (n being modulo)
        modulo = int(len(self.image_seq) / number_of_frames)
        if modulo == 0:
            modulo = 1
        for counter, img in enumerate(self.image_seq):
            if counter % modulo == 0 and index < number_of_frames:
                images.append(img)
                index += 1

        return images

    def valid_dataset(self, image_path, scaling, scale_x, scale_y):
        self.read_image_data(str(image_path), scaling=scaling, scale_x=scale_x, scale_y=scale_y)
        if len(self.image_seq) == 0:
            print("No image in %s" % (image_path))
            return False
        return True

    def read_image_data(self, data_dir, scaling='no scaling', scale_x=0.1, scale_y=0.1):
        data_dir += '/raw_images/'
        if scaling == 'scale':
            self.image_seq = self.load_images_from_folder(data_dir, scaling='scale', scale_x=scale_x, scale_y=scale_y)
        else:
            self.image_seq = self.load_images_from_folder(data_dir)
    
    def load_images_from_folder(self, folder, scaling='no scale', scale_x=0.1, scale_y=0.1):
        images = []
        filenames = sorted(os.listdir(folder))

        for filename in filenames:
            if self.valid_image(filename):
                img = cv2.imread(os.path.join(folder, filename)).astype(np.float32)
                img /= 255.0
                if img is not None:
                    if scaling == 'scale':
                        img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)
                    images.append(img)
        return images

    @staticmethod
    def rescale_images(source_dir, save_dir, scaling='scale', scale_x=0.1, scale_y=0.1):

        foldernames = [f for f in os.listdir(source_dir) if f.isnumeric() and not f.startswith('.')]

        for foldername in tqdm(foldernames):

            if foldername.isnumeric:
                newpath = save_dir + "/" + foldername
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                    for filename in os.listdir(source_dir + "/" + foldername):
                        img = cv2.imread(os.path.join(source_dir + "/" + foldername, filename))
                        if img is not None:
                            if scaling == 'scale':
                                img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)
                            cv2.imwrite(os.path.join(newpath, filename), img)
    
    def read_risk_data(self, parent_dir):
        risk_scores = []
        for clip in self.foldernames:
            path = parent_dir / clip
            label_path = path / "label.txt"
            if label_path.exists():
                with open(str(path/"label.txt"), 'r') as label_f:
                    risk_label = int(float(label_f.read().strip().split(",")[0]))
                    risk_scores.append(risk_label)
            else:
                raise FileNotFoundError("No label.txt in %s" % path) 
        return risk_scores

    def convert_risk_to_one_hot(self):
        # sorting risk thresholds from least risky to most risky
        indexes = [i[0] for i in sorted(enumerate(self.risk_scores), key=lambda x: x[1])]
        self.risk_one_hot = np.zeros([len(indexes), 2])

        for counter, index in enumerate(indexes[::-1]):
            if self.risk_scores[index] >= 0:
                self.risk_one_hot[index, :] = [0, 1]
            else:
                self.risk_one_hot[index, :] = [1, 0]

    # Utilities
    def get_clipnumber(self, elem):
        return int(elem.split('_')[0])

    def get_filtered_clips(self, clip_dir, foldernames):
        filtered_folders = []
        for foldername in foldernames:
            clip_path = clip_dir / foldername
            if self.ignore_clip(clip_path): continue;
            filtered_folders.append(foldername)

        return filtered_folders
    
    def get_max_frames(self, data_dir):
        '''
            Return the longest clip (by amount of frames)
            As well as the distribution of frames over all clips
            These clips are identifiable with dict clip_lookup
        '''
        clip_lookup = {}
        num_frames = []
        for foldername in self.foldernames:
            foldername = data_dir/foldername/'raw_images'
            imgs = [img for img in os.listdir(foldername) if self.valid_image(img)]
            num_frames.append(len(imgs))
            clip_lookup[foldername] = len(imgs)
        frame_dist = Counter(num_frames).most_common()
        # threshold
        # for key, val in clip_lookup.items():
        #     if val > 150:
        #         print(key)
        return max(num_frames), frame_dist, clip_lookup

    def ignore_clip(self, clip_dir):
        '''
            return 1 when ignore.txt is 1 (ignore clip
            return 0 when ignore.txt is 0 (do not ignore clip)
        '''
        ignore_path = clip_dir / 'ignore.txt'
        if ignore_path.exists():
             with open(str(ignore_path), 'r') as label_f:
                    ignore_label = int(label_f.read())
                    return ignore_label
        return False # no ignore.txt means to include it in dataset

    def save(self, filename, save_dir):
        with open(save_dir + filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def valid_image(self, filename):
        return Path(filename).suffix == '.jpg' or Path(filename).suffix == '.png'
