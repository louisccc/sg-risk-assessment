import os, sys
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from pprint import pprint
import wandb

import check_gpu as cg
os.environ['CUDA_VISIBLE_DEVICES'] = cg.get_free_gpu()

sys.path.append('../nagoya')
sys.path.append('../')
from nagoya.dataset import DataSet
from nagoya.models import LSTM_Classifier, CNN_LSTM_Classifier, CNN_Classifier, ResNet50_LSTM_Classifier
from nagoya.train import Trainer
# FIXME: Not working... ModuleNotFoundError: No module named 'prompt_toolkit.formatted_text'
# from nagoya.Mask_RCNN.mask_rcnn.detect_objects import DetectObjects

PROJECT_NAME = "Fill me with wandb id"

class Config:

	def __init__(self, args):
		self.parser = ArgumentParser(description='The parameters for configuring and training the baseline Nagoya model(s)')
		self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data", help="Path to data directory.")
		self.parser.add_argument('--coco_path', type=str, default="../pretrained_models/", help="Path to coco pretrained model.")
		self.parser.add_argument('--pkl_path', type=str, default="/home/louisccc/NAS/louisccc/av/nagoya_pkl_data/one_camera/5_frames_dataset.pkl", help="Path to pickled dataset.")
		self.parser.add_argument('--load_pkl', type=lambda x: (str(x).lower() == 'true'), default=False, help='Load model from cache.')
		self.parser.add_argument('--save_pkl_path', type=str, default="", help="Path to save pickled dataset.")
		self.parser.add_argument('--save_pkl', type=lambda x: (str(x).lower() == 'true'), default=False, help='Save pkl to save_pkl_path.')
		self.parser.add_argument('--model_name', type=str, default="cnn_lstm", help="Type of model to run, choices include [gru, lstm, cnn, cnn_lstm, resnet]")
		self.parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False, help='Load model from cache.')
		self.parser.add_argument('--model_path', type=str, default="../cache/RCNN_CNN_lstm_GPU_20_2.h5", help="Path to cached model file.")
		self.parser.add_argument('--mask_rcnn', type=lambda x: (str(x).lower() == 'true'), default=False, help='Create masked imgages.')

		# Training
		self.parser.add_argument('--n_folds', type=int, default=5, help="Number of cross validations")
		self.parser.add_argument('--train_ratio', type=float, default=0.7, help="Ratio of dataset used for testing")
		self.parser.add_argument('--downsample', type=lambda x: (str(x).lower() == 'true'), default=False, help='Downsample (balance) dataset.')
		self.parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
		self.parser.add_argument('--test_step', type=int, default=5, help='Number of training epochs before testing the model.')
		self.parser.add_argument('--device', type=str, default="cuda", help='The device on which models are run, options: [cuda, cpu].')
		self.parser.add_argument('--gradcam', type=lambda x: (str(x).lower() == 'true'), default=False, help='Compute gradcam over a single batch')

		# Hyperparameters
		self.parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train")
		self.parser.add_argument('--batch_size', type=int, default=32, help="Batch size per forward")		
		self.parser.add_argument('--bnorm', type=lambda x: (str(x).lower() == 'true'), default=False, help="Utilize batch normalization.")
		self.parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
		self.parser.add_argument('--learning_rate', default=3e-4, type=float, help='The initial learning rate.')
		self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

		args_parsed = self.parser.parse_args(args)
		
		self.wandb = wandb.init(project=PROJECT_NAME)
		self.wandb_config = self.wandb.config
		
		for arg_name in vars(args_parsed):
			self.__dict__[arg_name] = getattr(args_parsed, arg_name)
			self.wandb_config[arg_name] = getattr(args_parsed, arg_name)

		self.input_base_dir = Path(self.input_path).resolve()
		self.cache_model_path = Path(self.model_path).resolve()

def run_maskrcnn(config, root_folder_path, raw_image_path):
	coco_model_path = Path(config.coco_path) #Path('../pretrained_models')
	masked_image_path = root_folder_path / (raw_image_path.stem + '_masked') # the path in parallel with raw_image_path
	masked_image_path.mkdir(exist_ok=True)

	# Check if masked images already exist
	raw_folders = sorted([f for f in os.listdir(raw_image_path) 
													if f.split('_')[0].isnumeric() and not f.startswith('.')], 
													key=lambda x: int(x.split('_')[0]))
	masked_folders = sorted([f for f in os.listdir(masked_image_path) 
														 if f.split('_')[0].isnumeric() and not f.startswith('.')], 
														 key=lambda x: int(x.split('_')[0]))
	
	if len(raw_folders)!=len(masked_folders):
		process_raw_images_to_masked_images(raw_image_path, masked_image_path, coco_model_path)
	
	dataset = load_dataset(raw_image_path, masked_image_path, dataset_type="masked")
	return dataset

def process_raw_images_to_masked_images(src_path: Path, dst_path: Path, coco_path: Path):
    ''' 
        This step is for preprocessing the raw images 
        to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
    '''
    masked_image_extraction = DetectObjects(src_path, dst_path, coco_path)
    masked_image_extraction.save_masked_images()
	
def load_dataset(raw_image_path: Path, masked_image_path: Path, dataset_type: str, config=None):
	'''
		This step is for loading the dataset, preprocessing the video clips 
		and neccessary scaling and normalizing. Also it reads and converts the labeling info.
	'''
	image_path = masked_image_path if dataset_type == "masked" else raw_image_path

	dataset = DataSet()
	# dataset.read_video(image_path, option='all frames', scaling='scale', scale_x=0.05, scale_y=0.05)
	dataset.read_video(image_path, option='fixed frame amount', number_of_frames=5, scaling='scale', scale_x=0.05, scale_y=0.05)

	dataset.risk_scores = dataset.read_risk_data(raw_image_path)
	dataset.convert_risk_to_one_hot()

	if config != None and config.save_pkl:
		parent_path = '/'.join(config.save_pkl_path.split('/')[:-1]) + '/'
		fname = config.save_pkl_path.split('/')[-1]
		dataset.save(save_dir=parent_path, filename=fname)
		print("Saved pickled dataset")
	return dataset

def load_pickle(pkl_path: Path):
	'''
		Read dataset from pickle file.
	'''
	dataset = DataSet().loader(str(pkl_path))
	import pdb; pdb.set_trace()
	return dataset

def reshape_dataset(dataset):
	'''
		input -> (batch, frames, height, width, channels)
		output -> (batch, frames, channels, height, width)
	'''
	return np.swapaxes(np.swapaxes(dataset, -1, -3), -1, -2)

def train_model(dataset, config):
	dataset.video = reshape_dataset(dataset.video)
	video_sequences = dataset.video
	labels = dataset.risk_one_hot
	clip_names = np.array(['default_all']*len(video_sequences))
	if hasattr(dataset, 'foldernames'): 
		clip_names = np.concatenate((clip_names, dataset.foldernames), axis=0)

	if config.model_name == 'gru':
		model = LSTM_Classifier(video_sequences.shape, 'gru', config)
	elif config.model_name == 'lstm':
		model = LSTM_Classifier(video_sequences.shape, 'lstm', config)
	elif config.model_name == 'cnn':
		model = CNN_Classifier(video_sequences.shape, config)
	elif config.model_name == 'cnn_lstm':
		model = CNN_LSTM_Classifier(video_sequences.shape, config)
	elif config.model_name == 'resnet': 
		model = ResNet50_LSTM_Classifier(video_sequences.shape, config)
	
	trainer = Trainer(config)
	trainer.init_dataset(video_sequences, labels, clip_names)
	trainer.build_model(model)
	if config.n_folds > 1:
		trainer.train_n_fold_cross_val()
	else:
		trainer.train_model()
	
if __name__ == '__main__':
	config = Config(sys.argv[1:])
	raw_image_path = config.input_base_dir
	root_folder_path = raw_image_path.parent

	if config.load_pkl:
		dataset = load_pickle(Path(config.pkl_path).resolve())
	else:
		if config.mask_rcnn: dataset = run_maskrcnn(config, root_folder_path, raw_image_path);
		else: dataset = load_dataset(raw_image_path, raw_image_path, dataset_type="raw", config=config);
	
	# train model
	model = train_model(dataset, config)