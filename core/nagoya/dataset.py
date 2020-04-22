import pickle
import cv2
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
import pandas as pd
from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects
from pathlib import Path


class DataSet:

    def __init__(self):

        self.dataset = {}
        self.video = []
        self.image_seq = []
        self.risk_scres = []
        self.risk_one_hot = []
        self.risk_binary = []
        self.video_features = []
        self.feature_dataset = []
        self.model = []  # feature extractor

        self.can_seq = []
        self.image_parsed = []
        self.can_parsed = []

        self.num_frames, self.im_width, self.im_height = [], [], []

    def read_video(self, data_dir, option='fixed frame amount', number_of_frames=20, max_number_of_frames=500,
                   scaling='no scaling', scale_x=0.1, scale_y=0.1):

        foldernames = [f for f in sorted(os.listdir(data_dir),key=int) if f.isnumeric() and not f.startswith('.')]
        print(foldernames)
        #need to sort, global variable foldernames when read one hot match

        self.read_image_data(str(data_dir/foldernames[0]), scaling=scaling, scale_x=scale_x, scale_y=scale_y)
        
        if option == 'fixed frame amount':
            self.video = np.zeros([len(foldernames), number_of_frames, self.image_seq[000].shape[0],
                                   self.image_seq[000].shape[1], self.image_seq[000].shape[2]])
        elif option == 'all frames':
            self.video = np.zeros([len(foldernames), max_number_of_frames, self.image_seq[000].shape[0],
                                   self.image_seq[000].shape[1], self.image_seq[000].shape[2]])
            # shape: (n_vidoes, n_frames, im_height, im_width, channel)
        # todo convert this to a wrapper
        for foldername in tqdm(foldernames):
            if foldername.isnumeric:
                self.read_image_data(str(data_dir/foldername), scaling=scaling, scale_x=scale_x, scale_y=scale_y)
                if not len(self.image_seq) == 0:
                    if option == 'fixed frame amount':
                        self.video[int(foldername), :, :, :, :] = self._read_video_helper(number_of_frames=number_of_frames)
                    elif option == 'all frames':
                        self.video[int(foldername), 0:len(self.image_seq), :, :, :] = self.image_seq

    def _read_video_helper(self, number_of_frames=20):

        images = []
        index = 0
        
        # length of image sequence must be greater than or equal to number_of_frames
        # if number of frames is less than entire length of image sequence, takes every nth frame (n being modulo)
        modulo = int(len(self.image_seq) / number_of_frames)
        for counter, img in enumerate(self.image_seq):
            if counter % modulo == 0 and index < number_of_frames:
                images.append(img)
                index += 1

        return images

    def extract_features(self, img_path, feature_size=2048, option='fixed frame amount', number_of_frames=20,
                         max_number_of_frames=500):

        # looks for folders with numeric names and doesnt start with .
        foldernames = [f for f in os.listdir(img_path) if f.isnumeric() and not f.startswith('.')]
        int_foldernames = [int(f) for f in os.listdir(img_path) if f.isnumeric() and not f.startswith('.')]
   
        # fixed frame amount if specified to use only first n clips 
        if option == 'fixed frame amount':
            self.video_features = np.zeros([max(int_foldernames)+1, number_of_frames, feature_size])
        elif option == 'all frames':
            self.video_features = np.zeros([max(int_foldernames)+1, max_number_of_frames, feature_size])
            # shape: (n_vidoes, n_frames, im_height, im_width, channel)
        #shape of video_features is # of videos, # of frames, feature size

        # todo convert this to a wrapper
        # tqdm shows progress bar
        # image_seq has the features of the images
        for foldername in tqdm(foldernames):
            if foldername.isnumeric:
                self.image_seq = self.load_images_for_keras(img_path + "/" + foldername + "/raw_images")
                if not len(self.image_seq) == 0:
                    if option == 'fixed frame amount':
                        self.video_features[int(foldername), :, :] = self._read_video_helper(number_of_frames=number_of_frames)
                    elif option == 'all frames':
                        self.video_features[int(foldername), 0:len(self.image_seq), :] = self.image_seq

    def read_features(self, feature_path, feature_size=2048, option='fixed frame amount', number_of_frames=20,
                      max_number_of_frames=500):

        foldernames = [f for f in os.listdir(feature_path) if f.isnumeric() and not f.startswith('.')]
        int_foldernames = [int(f) for f in os.listdir(feature_path) if f.isnumeric() and not f.startswith('.')]
        if option == 'fixed frame amount':
            self.video_features = np.zeros([max(int_foldernames), number_of_frames, feature_size])
        elif option == 'all frames':
            self.video_features = np.zeros([max(int_foldernames), max_number_of_frames, feature_size])
            # shape: (n_vidoes, n_frames, im_height, im_width, channel)

        for foldername in tqdm(foldernames):
            if foldername.isnumeric:
                filenames = sorted(os.listdir(feature_path + '/' + foldername))
                index = 0

                for counter, filename in enumerate(filenames):
                    feature_file = feature_path + '/' + foldername + '/' + filename
                    # todo convert this to a wrapper
                    if option == 'fixed frame amount':
                        modulo = int(len(filenames) / number_of_frames)
                        if counter % modulo == 0 and index < number_of_frames:
                            self.video_features[int(foldername)-1, index, :] = np.loadtxt(feature_file, delimiter=',')
                            index += 1

                    elif option == 'all frames':
                        self.video_features[int(foldername) - 1, counter, :] = np.loadtxt(feature_file, delimiter=',')

    def read_image_data(self, data_dir, scaling='no scaling', scale_x=0.1, scale_y=0.1):

        if scaling == 'scale':
            self.image_seq = self.load_images_from_folder(data_dir, scaling='scale', scale_x=scale_x, scale_y=scale_y)

        else:
            self.image_seq = self.load_images_from_folder(data_dir)

    def read_can_data(self, data_dir):
        # todo
        self.can_seq = []

    def read_risk_data(self, file_path):
        df = pd.read_csv(file_path, header=None, usecols=[7], names=['risk_score'])
        #import pdb; pdb.set_trace()
        self.risk_scores = df['risk_score'].tolist()

    def convert_risk_to_one_hot(self, risk_threshold=0.5):
        # sorting risk thresholds from least risky to most risky
        indexes = [i[0] for i in sorted(enumerate(self.risk_scores), key=lambda x: x[1])]
        top_risky_threshold = int(len(indexes) * risk_threshold)
        self.risk_one_hot = np.zeros([len(indexes), 2])

        # assigning one hot vector [0,1] for risky and [1,0] for not risky
        # if risk threshold is 0.5, 50% risky (upper half of indexes array) 50% not risky (lower half of indexes array)
        for counter, index in enumerate(indexes[::-1]):
            if counter < top_risky_threshold:
                self.risk_one_hot[index, :] = [0, 1]
            else:
                self.risk_one_hot[index, :] = [1, 0]


    def decode_one_hot(self):
        self.risk_binary = np.zeros([self.risk_one_hot.shape[0], 1])
        self.risk_binary[:, 0] = np.argmax(self.risk_one_hot, axis=1)

    def save(self, filename='dataset.pickle', save_dir='saved data/'):
        with open(save_dir + filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def loader(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

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

    @staticmethod
    def load_images_from_folder(folder, scaling='no scale', scale_x=0.1, scale_y=0.1):

        images = []
        filenames = sorted(os.listdir(folder))

        for filename in filenames:

            img = cv2.imread(os.path.join(folder, filename)).astype(np.float32)
            img /= 255.0
            if img is not None:
                if scaling == 'scale':
                    img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)
                images.append(img)

        return images

    def load_images_for_keras(self, img_path, target_size=(224, 224)):

        features = []
        filenames = sorted(os.listdir(img_path))

        for filename in filenames:

            # load image with default target size 224(image height)x224(image width)
            # if image size differs from target size, uses nearest interpolation to resample image
            # to create a batch of images need an additional dimension so np.expand_dims
            # make image format compatible with model using preprocess_input
            # return features of image, object identification
            img = image.load_img(os.path.join(img_path, filename), target_size=target_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            feature = self.model.predict(img)
            if img is not None:
                features.append(feature)

        return features

def raw2masked(image_path, masked_image_path):
    ''' 
        This step is for preprocessing the raw images 
        to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
    '''
    masked_image_extraction = DetectObjects(image_path, masked_image_path)
    masked_image_extraction.save_masked_images()

def load_dataset(masked_image_path: Path, label_table_path: Path):
    '''
        This step is for loading the dataset, preprocessing the video clips 
        and neccessary scaling and normalizing. Also it reads and converts the labeling info.
    '''
    dataset = DataSet()
    dataset.read_video(masked_image_path, option='fixed frame amount', number_of_frames=5, scaling='scale', scale_x=0.1, scale_y=0.1)

    '''
        order videos by risk and find top riskiest
        #match input to risk label in LCTable 
        data = label_risk(masked_data)
    '''
    dataset.read_risk_data(str(label_table_path))
    dataset.convert_risk_to_one_hot(risk_threshold=0.5)

    return dataset