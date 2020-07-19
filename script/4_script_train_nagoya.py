import os, sys
from argparse import ArgumentParser
from pathlib import Path
import skimage.io as io
import matplotlib
matplotlib.use("Agg")

sys.path.append('../nagoya')
sys.path.append('../')
from nagoya.dataset import DataSet
from nagoya.models import Models
from nagoya.Mask_RCNN.mask_rcnn.detect_objects import DetectObjects
from core.dynkg_trainer import get_metrics
io.use_plugin('pil')


class Config:

	def __init__(self, args):
		self.parser = ArgumentParser(description='The parameters for creating gifs of input videos.')
		self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data", help="Path to data directory.")
		self.parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False, help='Load model from cache.')
		self.parser.add_argument('--model_path', type=str, default="../cache/RCNN_CNN_lstm_GPU_20_2.h5", help="Path to cached model file.")
		self.parser.add_argument('--mask_rcnn', type=lambda x: (str(x).lower() == 'true'), default=True, help='Create masked imgages.')
		args_parsed = self.parser.parse_args(args)
		
		for arg_name in vars(args_parsed):
			self.__dict__[arg_name] = getattr(args_parsed, arg_name)

		self.input_base_dir = Path(self.input_path).resolve()
		self.cache_model_path = Path(self.model_path).resolve()


def train_cnn_to_lstm(dataset, cache_path):
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
	model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
	model.build_cnn_to_lstm_model(input_shape=video_sequence.shape[1:])
	metrics = model.train_n_fold_cross_val(video_sequence, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0)
	
	cache_path.parent.mkdir(exist_ok=True)
	model.model.save(str(cache_path))
	print(metrics)
	return model

def process_raw_images_to_masked_images(src_path: Path, dst_path: Path, coco_path: Path):
    ''' 
        This step is for preprocessing the raw images 
        to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
    '''
    masked_image_extraction = DetectObjects(src_path, dst_path, coco_path)
    masked_image_extraction.save_masked_images()

def read_risk_data(masked_image_path: Path):
	risk_scores = []
	all_video_clip_dirs = [f for f in masked_image_path.iterdir() if f.is_dir() and f.stem.isnumeric()]
	all_video_clip_dirs = sorted(all_video_clip_dirs, key=lambda f: int(f.stem))
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
	dataset.read_video(image_path, option='all frames', number_of_frames=20, scaling='scale', scale_x=0.1, scale_y=0.1)

	'''
		order videos by risk and find top riskiest
		#match input to risk label in LCTable 
		data = label_risk(masked_data)
	'''
	dataset.risk_scores = read_risk_data(raw_image_path)
	dataset.convert_risk_to_one_hot(risk_threshold=0.5)

	return dataset

if __name__ == '__main__':
	
	config = Config(sys.argv[1:])
	
	raw_image_path = config.input_base_dir #Path('../input/synthesis_data').resolve()
	root_folder_path = raw_image_path.parent
	label_table_path = raw_image_path / "LCTable.csv"
	cache_model_path = config.cache_model_path
	
	do_mask_rcnn = config.mask_rcnn

	if do_mask_rcnn: 
		coco_model_path = root_folder_path / 'pretrained_models' #Path('../pretrained_models')
		masked_image_path = root_folder_path / (raw_image_path.stem + '_masked') # the path in parallel with raw_image_path
		masked_image_path.mkdir(exist_ok=True)

		#check if masked images already exist
		raw_folders = [f for f in os.listdir(raw_image_path) if f.isnumeric() and not f.startswith('.')]
		masked_folders = [f for f in os.listdir(masked_image_path) if f.isnumeric() and not f.startswith('.')]
		
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
		# model = load_model(str(cache_model_path))
	else:
		model = train_cnn_to_lstm(dataset, cache_model_path)
	
	'''
		determine how safe and dangerous each lane change is 
	'''
	true_label = np.argmax(dataset.risk_one_hot,axis=-1)
	end = int(0.7*len(dataset.video))
	output = model.predict_proba(dataset.video[end:])

	metrics = get_metrics(output,true_label[end:]) 
	print(metrics)
	print(' safe | dangerous \n', output)
