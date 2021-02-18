import sys, os
from pathlib import Path
from argparse import ArgumentParser

sys.path.append(os.path.dirname(sys.path[0]))


class Config:
    def __init__(self, args):
        self.parser = ArgumentParser(description="Parameters for extracting scenegraphs.")
        self.parser.add_argument('--input_path', type=str, default="/home/louisccc/NAS/louisccc/av/synthesis_data/new_recording_3", help="Path to lane-change clips directory.")
        self.parser.add_argument('--platform', type=str, default="carla", help="Method for scenegraph extraction (carla or image or honda).")
        self.parser.add_argument('--cache', type=lambda x: (str(x).lower() == 'true'), default=True, help="Cache processed scenegraphs.")
        self.parser.add_argument('--address', type=str, default="./image_dataset.pkl", help="Path to save cache file.")
        self.parser.add_argument('--visualize', type=lambda x: (str(x).lower() == 'true'), default=False, help="Visualize scenegraphs.")
        self.parser.add_argument('--vis_clipids', type=int, nargs='+', default=None, help='Folder Ids of the lane change clip to visualize.')
        self.parser.add_argument('--framenum', type=int, default=10, help='Number of frames to extract from each video clip.')

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
        generator = CarlaSceneGraphSequenceGenerator(cfg.framenum)
    elif cfg.platform == "image":
        from core.real_seq_generator import ImageSceneGraphSequenceGenerator
        generator = ImageSceneGraphSequenceGenerator(cfg.framenum)
    elif cfg.platform == "honda":
        from core.real_seq_generator import ImageSceneGraphSequenceGenerator
        generator = ImageSceneGraphSequenceGenerator(cfg.framenum, platform='honda')

    if cfg.visualize:
        generator.visualize_scenegraphs(cfg.vis_clipids)

    generator.load(cfg.input_base_dir)
    if cfg.cache:
        generator.cache_dataset(str(cfg.cache_path))

   