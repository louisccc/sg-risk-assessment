import os, sys
from argparse import ArgumentParser
from pathlib import Path
import skimage.io as io
import matplotlib
matplotlib.use("Agg")
import numpy as np 
import pandas as pd
from keras.models import load_model

sys.path.append('../nagoya')
sys.path.append('../')
from nagoya.dataset import DataSet
from nagoya.models import Models
from nagoya.Mask_RCNN.mask_rcnn.detect_objects import DetectObjects
from core.dynkg_trainer import get_metrics
from pprint import pprint
io.use_plugin('pil')


class Config:

	def __init__(self, args):
		self.parser = ArgumentParser(description='The parameters for creating gifs of input videos.')
		self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data", help="Path to data directory.")
		self.parser.add_argument('--coco_path', type=str, default="../pretrained_models/", help="Path to coco pretrained model.")
		self.parser.add_argument('--pkl_path', type=str, default="/home/louisccc/NAS/louisccc/av/nagoya_pkl_data/all_frames_dataset.pkl", help="Path to pickled dataset.")
		self.parser.add_argument('--load_pkl', type=lambda x: (str(x).lower() == 'true'), default=False, help='Load model from cache.')
		self.parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False, help='Load model from cache.')
		self.parser.add_argument('--model_path', type=str, default="../cache/RCNN_CNN_lstm_GPU_20_2.h5", help="Path to cached model file.")
		self.parser.add_argument('--mask_rcnn', type=lambda x: (str(x).lower() == 'true'), default=True, help='Create masked imgages.')
		self.parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
		self.parser.add_argument('--downsample', type=lambda x: (str(x).lower() == 'true'), default=False, help='Downsample dataset.')
		self.parser.add_argument('--stats_path', type=str, default="nagoya_best_stats.csv", help="Path to save best test statistics.")


		args_parsed = self.parser.parse_args(args)
		
		for arg_name in vars(args_parsed):
			self.__dict__[arg_name] = getattr(args_parsed, arg_name)

		self.input_base_dir = Path(self.input_path).resolve()
		self.cache_model_path = Path(self.model_path).resolve()

def save_metrics(metrics, config):
	filepath = Path(config.stats_path).resolve()

	best_metrics = {}
	best_metrics['seed'] = config.seed
	best_metrics['dataset'] = Path(config.pkl_path).name
	best_metrics['balanced'] = config.downsample
	best_metrics['epoch'] = metrics['epoch']
	best_metrics['val loss'] = metrics['test']['loss']
	best_metrics['val acc'] = metrics['test']['acc']
	best_metrics['val conf'] = metrics['test']['confusion']
	best_metrics['val auc'] = metrics['test']['auc']
	best_metrics['val precision'] = metrics['test']['precision']
	best_metrics['val recall'] = metrics['test']['recall']
	best_metrics['train loss'] = metrics['train']['loss']
	best_metrics['train acc'] = metrics['train']['acc']
	best_metrics['train conf'] = metrics['train']['confusion'] 
	best_metrics['train auc'] = metrics['train']['auc']
	best_metrics['train precision'] = metrics['train']['precision']
	best_metrics['train recall'] = metrics['train']['recall']
	
	if not filepath.exists():
		current_stats = pd.DataFrame(best_metrics, index=[0])
		current_stats.to_csv(str(filepath), mode='w+', header=True, index=False, columns=list(best_metrics.keys()))
	else:
		current_stats = pd.DataFrame(best_metrics, index=[0])
		current_stats.to_csv(str(filepath), mode='a', header=False, index=False, columns=list(best_metrics.keys()))

def train_cnn_to_lstm(dataset, cache_path, seed, downsample):
	'''
		This step is for training the CNN to LSTM model. (train from scratch architecture.)
	'''

	class_weight = {0: 0.05, 1: 0.95}
	training_to_all_data_ratio = 0.7
	nb_cross_val = 1
	nb_epoch = 200
	batch_size = 32

	end = int(0.7*len(dataset.video)) #training_to_all_data_ratio*len(dataset.video))
	video_sequence = dataset.video
	label = dataset.risk_one_hot
	model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
	model.build_cnn_to_lstm_model(input_shape=video_sequence.shape[1:])
	metrics = model.train_n_fold_cross_val(video_sequence, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0, seed=seed, downsample=downsample)
	
	cache_path.parent.mkdir(exist_ok=True)
	model.model.save(str(cache_path))
	return model, metrics

def eval_model(dataset, cache_path, seed, downsample):

	video_sequence = dataset.video
	label = dataset.risk_one_hot
	model = load_model(str(cache_path))

	true_label = np.argmax(label, axis=-1)
	y_pred_train = model.predict_proba(video_sequence)	
	metrics = get_metrics(y_pred_train, true_label)
	return model, metrics

def process_raw_images_to_masked_images(src_path: Path, dst_path: Path, coco_path: Path):
    ''' 
        This step is for preprocessing the raw images 
        to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
    '''
    masked_image_extraction = DetectObjects(src_path, dst_path, coco_path)
    masked_image_extraction.save_masked_images()

def read_risk_data(masked_image_path: Path):
	risk_scores = []
	all_video_clip_dirs = [f for f in masked_image_path.iterdir() if f.is_dir() and f.stem.split('_')[0].isnumeric()]
	all_video_clip_dirs = sorted(all_video_clip_dirs, key=lambda f: int(f.stem.split('_')[0]))
	for path in all_video_clip_dirs:
		label_path = path / "label.txt"
		if label_path.exists():
			with open(str(path/"label.txt"), 'r') as label_f:
				risk_label = int(float(label_f.read().strip().split(",")[0]))
				risk_scores.append(risk_label)
		else:
			raise FileNotFoundError("No label.txt in %s" % path) 
	return risk_scores
	
def load_dataset(raw_image_path: Path, masked_image_path: Path, dataset_type: str):
	'''
		This step is for loading the dataset, preprocessing the video clips 
		and neccessary scaling and normalizing. Also it reads and converts the labeling info.
	'''
	if dataset_type == "masked":
		image_path = masked_image_path
	else:
		image_path = raw_image_path

	dataset = DataSet()
	# dataset.read_video(image_path, option='all frames', number_of_frames=20, scaling='scale', scale_x=0.05, scale_y=0.05)
	dataset.read_video(image_path, option='fixed frame amount', number_of_frames=5, scaling='scale', scale_x=0.05, scale_y=0.05)

	'''
		order videos by risk and find top riskiest
		#match input to risk label in LCTable 
		data = label_risk(masked_data)
	'''
	dataset.risk_scores = read_risk_data(raw_image_path)
	dataset.convert_risk_to_one_hot(risk_threshold=0.5)
	save_dir = Path("/home/louisccc/NAS/louisccc/av/nagoya_pkl_data/").resolve()
	save_dir.mkdir(exist_ok=True)
	dataset.save(save_dir=str(save_dir), filename='/5_frames_nr4.pkl')
	print("Saved pickled dataset")
	return dataset

def load_pickle(pkl_path: Path):
	'''
		Read dataset from pickle file.
	'''
	dataset = DataSet().loader(str(pkl_path))
	return dataset

if __name__ == '__main__':
	
	config = Config(sys.argv[1:])
	
	raw_image_path = config.input_base_dir #Path('../input/synthesis_data').resolve()
	root_folder_path = raw_image_path.parent
	label_table_path = raw_image_path / "LCTable.csv"
	cache_model_path = config.cache_model_path
	
	do_mask_rcnn = config.mask_rcnn

	if config.load_pkl:
		dataset = load_pickle(Path(config.pkl_path).resolve())
	else:
		if do_mask_rcnn: 
			coco_model_path = Path(config.coco_path) #Path('../pretrained_models')
			masked_image_path = root_folder_path / (raw_image_path.stem + '_masked') # the path in parallel with raw_image_path
			masked_image_path.mkdir(exist_ok=True)

			#check if masked images already exist
			raw_folders = [f for f in sorted(os.listdir(raw_image_path), key=lambda x: int(x.split('_')[0])) if f.split('_')[0].isnumeric() and not f.startswith('.')]
			masked_folders = [f for f in sorted(os.listdir(masked_image_path), key=lambda x: int(x.split('_')[0])) if f.split('_')[0].isnumeric() and not f.startswith('.')]
			
			if len(raw_folders)!=len(masked_folders):
				process_raw_images_to_masked_images(raw_image_path, masked_image_path, coco_model_path)

			#load masked images
			dataset = load_dataset(raw_image_path, masked_image_path, dataset_type="masked")

		else:
			#load raw images
			dataset = load_dataset(raw_image_path, masked_image_path, dataset_type="raw")
	
	# load or train model
	if config.load_model:
		if not cache_model_path.exists():
			raise FileNotFoundError ("Cached model file not found.")
		metrics = eval_model(dataset, cache_model_path, config.seed, config.downsample)
		save_metrics(metrics, config)
	else:
		model, metrics = train_cnn_to_lstm(dataset, cache_model_path, config.seed, config.downsample)
		save_metrics(metrics, config)
