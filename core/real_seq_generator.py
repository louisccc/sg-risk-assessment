# -*- coding: utf-8 -*-
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import sys, os, pdb
import numpy as np
import cv2
import random
import networkx as nx
import itertools
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl
import json
import torch
import pandas as pd
from glob import glob
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import detectron2.utils.visualizer 
from detectron2.data import MetadataCatalog
sys.path.append(os.path.dirname(sys.path[0]))
from core.image_scenegraph import RealSceneGraph
from core.scene_graph import SceneGraph
from core.relation_extractor import ActorType, Relations, RELATION_COLORS
# from core.lane_extractor import LaneExtractor

# from enum import Enum

#SETTINGS FOR 1280x720 CARLA IMAGES:
CARLA_IMAGE_H = 720
CARLA_IMAGE_W = 1280
CROPPED_H = 350 #height of ROI. crops to lane area of carla images
BIRDS_EYE_IMAGE_H = 850 
BIRDS_EYE_IMAGE_W = 1280

H_OFFSET = CARLA_IMAGE_H - CROPPED_H #offset from top of image to start of ROI
Y_SCALE = 0.55 #18 pixels = length of lane line (10 feet)
X_SCALE = 0.54 #22 pixels = width of lane (12 feet)

CAR_PROXIMITY_THRESH_SUPER_NEAR = 50 # max number of feet between a car and another entity to build proximity relation
CAR_PROXIMITY_THRESH_VERY_NEAR = 150
CAR_PROXIMITY_THRESH_NEAR = 300
CAR_PROXIMITY_THRESH_VISIBLE = 500

LANE_THRESHOLD = 6 #feet. if object's center is more than this distance away from ego's center, build left or right lane relation
CENTER_LANE_THRESHOLD = 9 #feet. if object's center is within this distance of ego's center, build middle lane relation

def create_text_labels_with_idx(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{}_{} {:.0f}%".format(l, idx, s * 100) for idx, (l, s) in enumerate(zip(labels, scores))]
    return labels

detectron2.utils.visualizer._create_text_labels = create_text_labels_with_idx

#ROI: Region of Interest
#returns transformation matrix for warping image to birds eye projection
#birds eye matrix fixed for all images using the assumption that camera perspective does not change over time.
def get_birds_eye_matrix():
    src = np.float32([[0, CROPPED_H], [BIRDS_EYE_IMAGE_W, CROPPED_H], [0, 0], [BIRDS_EYE_IMAGE_W, 0]]) #original dimensions (cropped to ROI)
    dst = np.float32([[int(BIRDS_EYE_IMAGE_W*16/33), BIRDS_EYE_IMAGE_H], [int(BIRDS_EYE_IMAGE_W*17/33), BIRDS_EYE_IMAGE_H], [0, 0], [BIRDS_EYE_IMAGE_W, 0]]) #warped dimensions
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    #Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation (if needed)
    return M
    

#returns image warped to birds eye projection using M
#returned image is vertically cropped to the ROI (lane area)
def get_birds_eye_warp(image_path, M):
    img = cv2.imread(image_path)
    img = img[H_OFFSET:CARLA_IMAGE_H, 0:BIRDS_EYE_IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (BIRDS_EYE_IMAGE_W, BIRDS_EYE_IMAGE_H)) # Image warping
    warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB) #set to RGB
    return warped_img


class ImageSceneGraphSequenceGenerator:
    def __init__(self, cache_fname='real_dyngraph_embeddings.pkl'):
        # [ 
        #   {'node_embeddings':..., 'edge_indexes':..., 'edge_attrs':..., 'label':...}  
        # ]
        self.scenegraphs_sequence = []

        # cache_filename determine the name of caching file name storing self.scenegraphs_sequence and 
        self.cache_filename = cache_fname

        # flag for turning on visualization
        self.visualize = False
        
        # config used for parsing CARLA:
        # this is the number of global classes defined in CARLA.
        self.num_classes = 8
        
        # gets a list of all feature labels (which will be used) for all scenegraphs
        self.feature_list = {"rel_location_x", 
                             "rel_location_y", #add 3 columns for relative vector values
                             "distance_abs", # adding absolute distance to ego
                            }
        # create 1hot class labels columns.
        for i in range(self.num_classes):
            self.feature_list.add("type_"+str(i))
        
        # detectron setup. 
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.coco_class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
        self.predictor = DefaultPredictor(self.cfg)

    def cache_exists(self):
        return Path(self.cache_filename).exists()

    def load_from_cache(self):
        with open(self.cache_filename,'rb') as f: 
            self.scenegraphs_sequence , self.feature_list = pkl.load(f)

    def load(self, input_path):
        all_video_clip_dirs = [x for x in input_path.iterdir() if x.is_dir()]

        for path in tqdm(all_video_clip_dirs):
            scenegraphs = {} 
            raw_images = sorted(list(path.glob("raw_images/*.jpg")) + list(path.glob("raw_images/*.png")), key=lambda x: int(x.stem))
            for raw_image_path in raw_images:
                frame = raw_image_path.stem
                ### get bounding boxes using detectron. 
                out_img_path = None
                if self.visualize:
                    out_img_path = Path(raw_image_path).resolve().parent.parent / "obj_det_results"
                    out_img_path.mkdir(exist_ok=True)
                    out_img_path = str(out_img_path / str(Path(raw_image_path).name))
                bounding_boxes = self.get_bounding_boxes(str(raw_image_path),out_img_path=out_img_path)
                ### get lane prediction using lanenet.
                ### use two information to generate the corresponding scenegraphs.
                scenegraph = RealSceneGraph(str(raw_image_path), bounding_boxes, coco_class_names=self.coco_class_names)
                scenegraphs[frame] = scenegraph
            
            label_path = (path/"label.txt").resolve()

            if label_path.exists():
                with open(str(path/"label.txt"), 'r') as label_f:
                    risk_label = float(label_f.read().strip().split(",")[0])

                if risk_label >= 0:
                    risk_label = 1
                else:
                    risk_label = 0

                # scenegraph_dict contains node embeddings edge indexes and edge attrs.
                scenegraphs_dict = {}
                subsampled_scenegraphs, frame_numbers = self.subsample(scenegraphs, 1000)
                scenegraphs_dict['sequence'] = self.process_graph_sequences(subsampled_scenegraphs, frame_numbers, folder_name=path.name)
                scenegraphs_dict['label'] = risk_label
                scenegraphs_dict['folder_name'] = path.name
                  
                if self.visualize:
                    vis_folder_name = path / "image_visualize"
                    print("writing scenegraphs to %s"% str(vis_folder_name))
                    # if vis_folder_name.exists():
                    #     shutil.rmtree(vis_folder_name)
                    for scenegraph, frame_number in zip(subsampled_scenegraphs, frame_numbers): 
                        vis_folder_name.mkdir(exist_ok=True)
                        scenegraph.visualize(to_filename=str(vis_folder_name / "{}.png".format(frame_number)))

                self.scenegraphs_sequence.append(scenegraphs_dict)
            else:
                raise Exception("no label.txt in %s" % path) 
    
    def cache_dataset(self, filename):
        with open(str(filename), 'wb') as f:
            pkl.dump((self.scenegraphs_sequence, self.feature_list), f)
    
    def get_bounding_boxes(self, img_path, out_img_path=None):
        im = cv2.imread(img_path)
        outputs = self.predictor(im)

        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)
        # import pdb; pdb.set_trace()

        if out_img_path:
            # We can use `Visualizer` to draw the predictions on the image.
            v = detectron2.utils.visualizer.Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(out_img_path, out.get_image()[:, :, ::-1])

        return outputs["instances"].pred_boxes, outputs["instances"].pred_classes, outputs["instances"].image_size
            
    def process_graph_sequences(self, scenegraphs, frame_numbers, folder_name=None):
        '''
            The self.scenegraphs_sequence should be having same length after the subsampling. 
            This function will get the graph-related features (node embeddings, edge types, adjacency matrix) from scenegraphs.
            in tensor formats.
        '''
        sequence = []

        for idx, (scenegraph, frame_number) in enumerate(zip(scenegraphs, frame_numbers)):
            sg_dict = {}
            
            node_name2idx = {node:idx for idx, node in enumerate(scenegraph.g.nodes)}

            sg_dict['node_features']                    = self.get_node_embeddings(scenegraph)
            sg_dict['edge_index'], sg_dict['edge_attr'] = self.get_edge_embeddings(scenegraph, node_name2idx)
            sg_dict['folder_name'] = folder_name
            sg_dict['frame_number'] = frame_number
            sg_dict['node_order'] = node_name2idx
            sequence.append(sg_dict)
            
        # import pdb; pdb.set_trace()
        return sequence
    
    def visualize_scenegraphs(self):
        self.visualize = True

    def subsample(self, scenegraphs, number_of_frames=20): 
        '''
            This function will subsample the original scenegraph sequence dataset (self.scenegraphs_sequence). 
            Before running this function, it includes a variant length of graph sequences. 
            We expect the length of graph sequences will be homogenenous after running this function.

            The default value of number_of_frames will be 20; Could be a tunnable hyperparameters.
        '''
        sequence = []
        frame_numbers = []
        acc_number = 0
        modulo = int(len(scenegraphs) / number_of_frames)
        if modulo == 0:
            modulo = 1

        for idx, (timeframe, scenegraph) in enumerate(scenegraphs.items()):
            if idx % modulo == 0 and acc_number < number_of_frames:
                sequence.append(scenegraph)
                frame_numbers.append(timeframe)
                acc_number+=1
    
        return sequence, frame_numbers
        
    def get_node_embeddings(self, scenegraph):
        rows = []
        labels=[]
        ego_attrs = None
        
        #extract ego attrs for creating relative features
        for node, data in scenegraph.g.nodes.items():
            if "ego" in str(node).lower():
                ego_attrs = data['attr']
           
        if ego_attrs == None:
            raise NameError("Ego not found in scenegraph")
          
        def get_embedding(node, row):
            for key in self.feature_list:
                if key in node.attr:
                    row[key] = node.attr[key]
            row['type_'+str(node.label.value)] = 1 #assign 1hot class label
            return row
        
        for idx, node in enumerate(scenegraph.g.nodes):
            d = defaultdict()
            row = get_embedding(node, d)
            labels.append(node.label.value)
            rows.append(row)
            
        embedding = pd.DataFrame(data=rows, columns=self.feature_list)
        embedding = embedding.fillna(value=0) #fill in NaN with zeros
        embedding = torch.FloatTensor(embedding.values)
        
        return embedding

    def get_edge_embeddings(self, scenegraph, node_name2idx):
        edge_index = []
        edge_attr = []
        for src, dst, edge in scenegraph.g.edges(data=True):
            edge_index.append((node_name2idx[src], node_name2idx[dst]))
            edge_attr.append(edge['object'].value)
        
        edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
        edge_attr  = torch.LongTensor(edge_attr)
        
        return edge_index, edge_attr