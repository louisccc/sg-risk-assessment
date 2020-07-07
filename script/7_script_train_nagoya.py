import sys
sys.path.append('../nagoya')
sys.path.append('../')
from nagoya.dataset import *
from nagoya.models import Models
from argparse import ArgumentParser
from pathlib import Path
import skimage.io as io

io.use_plugin('pil')
class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for creating gifs of input videos.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data", help="Path to data directory.")

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()


def train_cnn_to_lstm(dataset):
	'''
		This step is for training the CNN to LSTM model. (train from scratch architecture.)
	'''

	class_weight = {0: 0.05, 1: 0.95}
	training_to_all_data_ratio = 0.7
	nb_cross_val = 1
	nb_epoch = 100
	batch_size = 32

	end = int(0.7*len(dataset.video)) #training_to_all_data_ratio*len(dataset.video))
	video_sequence = dataset.video
	label = dataset.risk_one_hot
	#import pdb;pdb.set_trace()
	model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
	model.build_cnn_to_lstm_model(input_shape=video_sequence.shape[1:])
	metrics = model.train_n_fold_cross_val(video_sequence, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0)
	
	# storing the model weights to cache folder.
	cache_folder = Path('../cache').resolve()
	cache_folder.mkdir(exist_ok=True)
	print(cache_folder)
	#import pdb; pdb.set_trace()
	model.model.save(str(cache_folder / '804_maskRCNN_CNN_lstm_GPU_20_2.h5'))
	print(metrics)
	import pdb;pdb.set_trace()

def process_raw_images_to_masked_images(src_path: Path, dst_path: Path, coco_path: Path):
    ''' 
        This step is for preprocessing the raw images 
        to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
    '''
    masked_image_extraction = DetectObjects(src_path, dst_path, coco_path)
    masked_image_extraction.save_masked_images()


if __name__ == '__main__':
	
	config = Config(sys.argv[1:])
	
	root_folder_path = config.input_base_dir #Path('../input/synthesis_data').resolve()
	raw_image_path = root_folder_path / 'lane-change-804'
	label_table_path = raw_image_path / "LCTable.csv"
	
	do_mask_rcnn = True

	if do_mask_rcnn: 
		coco_model_path = config.input_base_dir / 'pretrained_models' #Path('../pretrained_models')
		masked_image_path = root_folder_path / (raw_image_path.stem + '_masked') # the path in parallel with raw_image_path
		masked_image_path.mkdir(exist_ok=True)

		#check if masked images already exist
		raw_folders = [f for f in os.listdir(raw_image_path) if f.isnumeric() and not f.startswith('.')]
		masked_folders = [f for f in os.listdir(masked_image_path) if f.isnumeric() and not f.startswith('.')]
		
		print(raw_folders,masked_folders)
		if len(raw_folders)!=len(masked_folders):
		#if raw_folders[-1]!=masked_folders[-1]:
			process_raw_images_to_masked_images(raw_image_path, masked_image_path, coco_model_path)
		
		#load masked images
		dataset = load_dataset(masked_image_path, label_table_path)

	else:
		#load raw images
		dataset = load_dataset(raw_image_path, label_table_path)
	
	# train import from core
	# output store in maskRCNN_CNN_lstm_GPU.h5
	# use prediction script to evaluate
	#import pdb; pdb.set_trace()

	train_cnn_to_lstm(dataset)
