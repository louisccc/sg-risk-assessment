import sys, os
from pathlib import Path
from argparse import ArgumentParser

sys.path.append(os.path.dirname(sys.path[0]))


class Config:
    def __init__(self, args):
        self.parser = ArgumentParser(description="Parameters for extracting scenegraphs.")
        self.parser.add_argument('--input_path', type=str, default="/home/aung/NAS/louisccc/av/synthesis_data/new_recording_3", help="Path to lane-change clips directory.")
        self.parser.add_argument('--platform', type=str, default="image", help="Method for scenegraph extraction (carla or image).")

        args_parsed = self.parser.parse_args(args)
            
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
        self.input_base_dir = Path(self.input_path).resolve()

if __name__ == '__main__':
    cfg = Config(sys.argv[1:])
    if cfg.platform == "carla":
        from core.scene_graph import CarlaSceneGraphSequenceGenerator
        generator = CarlaSceneGraphSequenceGenerator()
        generator.load(cfg.input_base_dir)
        generator.cache_dataset("carla_dataset.pkl")

    elif cfg.platform == "image":
        from core.image_scenegraph import ImageSceneGraphSequenceGenerator
        generator = ImageSceneGraphSequenceGenerator()
        generator.load(cfg.input_base_dir)
        generator.cache_dataset("image_dataset.pkl")