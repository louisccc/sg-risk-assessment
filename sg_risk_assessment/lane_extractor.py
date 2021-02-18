import pickle as pkl
from collections import defaultdict
import os



class LaneExtractor:
    def __init__(self, image_dir=None):
        if not image_dir == None:
            self.cached_dir = image_dir
            self.cached_lanes = self.load_dict(image_dir)


    #get lane dict from directory
    def load_dict(self, image_dir):
        filepath = os.path.join(image_dir + "lanedicts.pkl")
        if not os.path.exists(filepath):
            print("No lanes found for dir: "+str(image_dir))
            self.cached_lanes = None
            self.cached_dir = image_dir
            return

        with open(filepath, 'rb') as f:
            lanedict = pkl.load(f)
        
        self.cached_lanes = lanedict
        self.cached_dir = image_dir

        return lanedict

    #return a dictionary containing lane masks for a single image
    #returns None if no lanes were detected for that file/directory
    def get_lanes_from_file(self, filepath):
        file_dir, filename = os.path.split(filepath)
        if file_dir == self.cached_dir:
            return self.cached_lanes[filename] if self.cached_lanes != None else None
        else:
            self.load_dict(file_dir)
            return self.cached_lanes[filename] if self.cached_lanes != None else None
