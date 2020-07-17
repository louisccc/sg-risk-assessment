import sys, os
from pathlib import Path
from argparse import ArgumentParser

sys.path.append(os.path.dirname(sys.path[0]))


class Config:
    def __init__(self, args):
        self.parser = ArgumentParser(description="Parameters for extracting scenegraphs.")
        self.parser.add_argument('--input_path', type=str, default="/home/aung/NAS/louisccc/av/synthesis_data/new_recording_3", help="Path to lane-change clips directory.")
        self.parser.add_argument('--platform', type=str, default="image", help="Method for scenegraph extraction (carla or image).")
        self.parser.add_argument('--cache', type=lambda x: (str(x).lower() == 'true'), default=True, help="Cache processed scenegraphs.")
        self.parser.add_argument('--address', type=str, default="./image_dataset.pkl", help="Path to save cache file.")
        self.parser.add_argument('--visualize', type=lambda x: (str(x).lower() == 'true'), default=False, help="Visualize scenegraphs.")
        args_parsed = self.parser.parse_args(args)
            
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
        self.input_base_dir = Path(self.input_path).resolve()
        self.cache_path = Path(self.address).resolve()

if __name__ == '__main__':
    cfg = Config(sys.argv[1:])
    generator = None

    if cfg.platform == "carla":
        from core.carla_seq_generator import CarlaSceneGraphSequenceGenerator
        generator = CarlaSceneGraphSequenceGenerator()
    elif cfg.platform == "image":
        from core.real_seq_generator import ImageSceneGraphSequenceGenerator
        generator = ImageSceneGraphSequenceGenerator()

    if cfg.visualize:
        generator.visualize_scenegraphs()

    generator.load(cfg.input_base_dir)
    if cfg.cache:
        generator.cache_dataset(str(cfg.cache_path))

   