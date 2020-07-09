import sys, pdb, os
sys.path.append(os.path.dirname(sys.path[0]))
from pathlib import Path
from core.scene_graph.scene_graph import CarlaSceneGraphSequenceGenerator
from core.detectron import ImageSceneGraphSequenceGenerator
import json


if __name__ == '__main__':
    src_path = Path("/home/aung/NAS/louisccc/av/synthesis_data/new_recording_3")
    if platform == "carla":
        generator = CarlaSceneGraphSequenceGenerator()
        generator.load(src_path)
        generator.cache_dataset("carla_dataset.pkl")

    elif platform == "image":
        generator = ImageSceneGraphSequenceGenerator()
        generator.load(src_path)
        generator.cache_dataset("image_dataset.pkl")