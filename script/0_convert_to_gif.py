import imageio
from pathlib import Path
import sys, os
from argparse import ArgumentParser
from tqdm import tqdm

class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for creating gifs of input videos.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data/lane-change/0", help="Path to data directory.")
        self.parser.add_argument('--recursive', type=lambda x: (str(x).lower() == 'true'), default=False, help='Recursive loading gifs')

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()



def convert_gif(path):
	folder_path = path / 'raw_images'
	img_path = folder_path.glob('**/*.jpg')
	images = []
	exists = os.path.isfile(path / 'lane_change.gif')

	if (not exists):
		for filename in img_path:
			images.append(imageio.imread(str(filename)))
		imageio.mimsave(path / 'lane_change.gif', images, format='GIF')


if __name__ == "__main__":
	config = Config(sys.argv[1:])

	if config.recursive:
		input_path = Path(os.path.split(config.input_base_dir)[0]).resolve()
		foldernames = [f for f in sorted(os.listdir(input_path)) if f.isnumeric()]
		foldernames = sorted(foldernames,key=int)

		for foldername in tqdm(foldernames):
			gif_path = input_path / foldername
			convert_gif(gif_path)
	else:
		print(config.input_base_dir)
		convert_gif(config.input_base_dir)
