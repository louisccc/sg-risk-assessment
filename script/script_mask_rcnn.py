import sys
sys.path.append('../core')
from nagoya.dataset import *
from nagoya.models import Models
from argparse import ArgumentParser
from pathlib import Path

class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for creating gifs of input videos.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data", help="Path to data directory.")
        self.parser.add_argument('--coco_path', type=str, default="../pretrained_models", help="Path to cache h5 file's directory.")

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()
        self.coco_path = Path(self.coco_path).resolve()
	

def call_mask_rcnn(src_path: Path, dst_path: Path, coco_path: Path):
    ''' 
        This step is for preprocessing the raw images 
        to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
    '''
    masked_image_extraction = DetectObjects(src_path, dst_path, coco_path)
    masked_image_extraction.save_masked_images()


if __name__ == '__main__':
	
	config = Config(sys.argv[1:])
	
	root_folder_path = config.input_base_dir #Path('../input/synthesis_data').resolve()
	raw_image_path = root_folder_path / 'lane-change'
	label_table_path = raw_image_path / "LCTable.csv"
	
	do_mask_rcnn = True

	if do_mask_rcnn: 
		coco_model_path = config.coco_path
		masked_image_path = root_folder_path / (raw_image_path.stem + '_masked') # the path in parallel with raw_image_path
		masked_image_path.mkdir(exist_ok=True)

		call_mask_rcnn(raw_image_path, masked_image_path, coco_model_path)
		
		#load masked images
		dataset = load_dataset(masked_image_path, label_table_path)

	else:
		#load raw images
		dataset = load_dataset(raw_image_path, label_table_path)
	
	# train import from core
	# output store in maskRCNN_CNN_lstm_GPU.h5
	# use prediction script to evaluate
	train_cnn_to_lstm(dataset)
