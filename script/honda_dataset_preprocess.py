import argparse
import glob
import os
import pickle as pkl
import subprocess
import threading
import time
from os import path

import h5py
import numpy as np
from datetime import datetime
from tqdm import tqdm

from pathlib import Path
import numpy as np
import sys
import pickle as pkl
import cv2
import pprint


class DataIndex:
    def __init__(self, root, annotation_dir, cache_dir, index_path):
        """
        arguments

        root:            string, path to the dataset
        annotation_dir: string, path to dir with eaf files
        """
        if index_path.exists():
            cache_data = pkl.load(open(str(index_path), 'rb'))

            self.root = cache_data['root']
            self.annotation_dir = cache_data['annotation_dir']
            self.events = cache_data['events']
            self.events_pd = cache_data['events_pd']
            self.layer_ix = cache_data['layer_ix']
            self.event_type_ix  = cache_data['event_type_ix']
            self.inv_layer_ix = cache_data['inv_layer_ix']
        else:
            # parse all data
            self.root = root
            self.annotation_dir = annotation_dir
            self.cache_dir = cache_data
            self.events, _ = self.get_all_session_data()
            # data indexing
            # data contains the same event_id in different layers!
#            print(self.events)
            self.events_pd, self.layer_ix, self.event_type_ix = self.convert_to_dataframe(self.events)

            """
            Three different layers are listed in the EAF files
            
            u'Cause'
            u'Operation_Stimuli-driven'
            u'Operation_Goal-oriented'
            """
            self.inv_layer_ix = {v: k for k, v in self.layer_ix.items()}

            #inst = pkl.load(open("saved_index.pkl", 'rb'))
            # monkey solution because something is wrong pickle/__new__/__getnewargs__
            cache_data = {
                'root': self.root,
                'annotation_dir': self.annotation_dir,
                'events': self.events,
                'events_pd': self.events_pd,
                'layer_ix': self.layer_ix,
                'event_type_ix': self.event_type_ix,
                'inv_layer_ix': self.inv_layer_ix
            }
            print(cache_data)
            pkl.dump(cache_data, open(self.cache_dir + "saved_index.pkl", 'wb'))

        # self.inv_event_type_ix = {v: k for k, v in self.event_type_ix.iteritems()}

    def get_all_session_data(self):
        events = None
        annotation_files = glob.glob(self.annotation_dir + "*.eaf")
        session_num = 0
        for annotation_file in annotation_files:
            print(annotation_file)
            try:
                events = self.collect_events_from_eaf(annotation_file, events)
                session_num += 1
            except KeyError:
                print("Error while parsing '{0}'".format(annotation_file))
                with open("corrupted_sessions.txt", 'a') as dump:
                    dump.write(annotation_file)
        print("Processed {0} sessions".format(session_num))
        return events, session_num


    def collect_events_from_eaf(self, input_file, append_to=None):
        if append_to:
            events = append_to
        else:
            events = {}
        eafob = pympi.Elan.Eaf(input_file)
        session_id = "".join(input_file.split("/")[-1].split("-")[:-1])
        layers = eafob.get_tier_names()
        
        for layer in layers:
            if layer not in events:
                events[layer] = {}
            for annotation in eafob.get_annotation_data_for_tier(layer):
                stripped_label = annotation[2].strip()
                if stripped_label not in events[layer]:
                    events[layer][stripped_label] = []
                events[layer][stripped_label].append((session_id, annotation[0], annotation[1]))
        return events


    def convert_to_dataframe(self, events):
        all_data = []
        layers = {}
        layer_ix = 0
        event_types = {}
        event_type_ix = 0
        for layer in events:
            layers[layer_ix] = layer
            for event_type in events[layer]:
                event_types[event_type_ix] = event_type
                for event in events[layer][event_type]:
                    # TODO: probably should construct the path here for simplicity
                    all_data.append(
                        [layer_ix, event_type_ix, event[0], event[1], event[2]]
                        )
                event_type_ix += 1
            layer_ix += 1

        #adds extra empty event
        event_types[len(event_types.items())] = "empty"
        dataframe = pd.DataFrame(data=all_data, columns=["layer", "event_type", "session_id",
                                                         "start", "end"])
        return dataframe, layers, event_types


"""
Deafult configuration for models
"""

class GeneralCfg:
    def __init__(self):
        #self.id = "GeneralSensors"
        # ############## Global Parameters ##############
        self.root = Path(r"/home/aung/NAS/louisccc/av/honda_data/release_2019_01_20")
        self.session_template = "{0}/{1}_{2}_{3}_ITS1/{4}/"
        self.annotation_dir = self.root / "EAF"
        self.sampling_frequency = 3
        self.video_framerate = 30

        self.extracted_features_dir = Path(r"/home/aung/NAS/louisccc/av/honda_data/cache/HRI_final_release")
        self.cache_dir = Path(r"/home/aung/NAS/louisccc/av/honda_data/cache/HRI_final_release")
        self.cache_format = "npy"
        self.cache_precision = "fp16"

        self.index_path = Path(r"/home/aung/NAS/louisccc/av/honda_data/EAF_parsing/saved_index.pkl")

        # set final split here
        self.train_session_set = [
            '201702271017', '201702271123', '201702271136', '201702271438',
            '201702271632', '201702281017', '201702281511', '201702281709',
            '201703011016', '201703061033', '201703061107', '201703061323',
            '201703061353', '201703061418', '201703061429', '201703061456',
            '201703061519', '201703061541', '201703061606', '201703061635',
            '201703061700', '201703061725', '201703080946', '201703081008',
            '201703081055', '201703081152', '201703081407', '201703081437',
            '201703081509', '201703081549', '201703081617', '201703081653',
            '201703081723', '201703081749', '201704101354', '201704101504',
            '201704101624', '201704101658', '201704110943', '201704111011',
            '201704111041', '201704111138', '201704111202', '201704111315',
            '201704111335', '201704111402', '201704111412', '201704111540',
            '201706061021', '201706070945', '201706071021', '201706071319',
            '201706071458', '201706071518', '201706071532', '201706071602',
            '201706071620', '201706071630', '201706071658', '201706071735',
            '201706071752', '201706080945', '201706081335', '201706081445',
            '201706081626', '201706081707', '201706130952', '201706131127',
            '201706131318', '201706141033', '201706141147', '201706141538',
            '201706141720', '201706141819', '201709200946', '201709201027',
            '201709201221', '201709201319', '201709201530', '201709201605',
            '201709201700', '201709210940', '201709211047', '201709211317',
            '201709211444', '201709211547', '201709220932', '201709221037',
            '201709221238', '201709221313', '201709221435', '201709221527',
            '201710031224', '201710031247', '201710031436', '201710040938',
            '201710060950', '201710061114', '201710061311', '201710061345',
        ]
        # this validation has 1 U-turn
        self.validation_session_set = [
            '201704101118', '201704130952', '201704131020', '201704131047',
            '201704131123', '201704131537', '201704131634', '201704131655',
            '201704140944', '201704141033', '201704141055', '201704141117',
            '201704141145', '201704141243', '201704141420', '201704141608',
            '201704141639', '201704141725', '201704150933', '201704151035',
            '201704151103', '201704151140', '201704151315', '201704151347',
            '201704151502', '201706061140', '201706061309', '201706061536',
            '201706061647', '201706140912', '201710031458', '201710031645',
            '201710041102', '201710041209', '201710041351', '201710041448',
        ]

        
    #======================================================================
    # ############## Feature Extraction Dirs ##############
        # cache_dir and extracted_features_dir currently have the same purpose and should match.
        # originally "feature_extraction.py" was used to extract LSTM features for analysis.
        # self.extracted_features_dir = self.root + "cache/HRI_final_release/"
        # self.cache_dir = self.root + "cache/HRI_final_release/"
        # self.cache_format = "npy"
        # self.cache_precision = "fp16"

        # temp for feature_extraction.py (should match self.cache_dir)
        # ############# Model saving parameters ##############
        self.logdir = "logs/"
        self.max_to_keep = 100  # how many ckpt files to keep while training (most recent ones)

        # ############# Miscellaneous ##############
        self.log_verbosity = "Error"  # disables warnings from TensorFlow
        self.cnn_checkpoint = 'experiments/slim/models/inception_resnet_v2_2016_08_30.ckpt'
        self.slim_dir = "/scratch4/repos/models/research/slim/"
        # frame sampling frequency; not tested with other values, some parts of the code assume = 3

        self.jobs = 2                                                          # number of readers during LSTM training
    #======================================================================

def print_metadata(cache_data):
    # layer_ix is the 4-layer representation proposed in the paper
    # Goal: 0, Stimulus: 6; Cause: 1; Attention: 3, 5; the rest are used for additional note: 2, 4
    pprint.pprint(cache_data['layer_ix'])
    
    # event_type_ix is all the categories defined in the 4-layer representation
    # Event_type_ix 0~12 belongs to those events defined in "Goal-oriented" layer. Note that we exclude 9 from the experiments conducted in CVPR'18.
    # Event type_ix 16 (congestion), 17 (sign), 18 (red light), 19 (crossing vehicle), 20 (parked vehicle), 22 (crossing pedestrian) 
    # are those objects annotated in "Cause" layer 
    pprint.pprint(cache_data['event_type_ix'])

def get_video_path(session_id):
    session_template = cfg.session_template  # "{0}/{1}_{2}_{3}_ITS_data_collection/{4}_ITS/"
    preview_template = session_template.format(cfg.root,
                                            session_id[:4],
                                            session_id[4:6],
                                            session_id[6:8],
                                            session_id) + "camera/center/*.mp4"
    video_full_path = glob.glob(preview_template)[0]
    
    return video_full_path


if __name__ == "__main__":
    
    print("Configuration...")
    cfg = GeneralCfg()
    index = DataIndex(cfg.root, cfg.annotation_dir, cfg.cache_dir, cfg.index_path)
    print("Total number of data records: {}".format(index.events_pd.shape[0]))

    cache_data = pkl.load(open(str(cfg.index_path), 'rb'))
    # print_metadata(cache_data)

    dest = Path('/home/aung/NAS/louisccc/av/honda_data/lane-change-clips').resolve()

    event_types = [3, 5]

    events = cache_data['events_pd'].loc[cache_data['events_pd']["event_type"].isin(event_types)]

    fps = cfg.video_framerate
    frame_padding = 30
    for idx, row in events.iterrows():
        start, end = int(row["start"] / 1000 * fps), int(row["end"] / 1000 * fps)
        session_id = row['session_id']

        out_path =  dest / (str(idx)+"_"+ session_id)
        out_path.mkdir(exist_ok=True)

        video_path = get_video_path(session_id)

        cap = cv2.VideoCapture(video_path)
        for frame_no in range(start - frame_padding, end + frame_padding, cfg.sampling_frequency):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = out_path /  (str(frame_no)+'.jpg')
            frame = cv2.resize(frame,(1024,1024))
            cv2.imwrite(str(outname), frame)
        cap.release()