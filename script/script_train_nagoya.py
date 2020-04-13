import sys
sys.path.append('../core')
from nagoya.dataset import *
from nagoya.models import Models

from pathlib import Path

def train_cnn_to_lstm(dataset):
	'''
		This step is for training the CNN to LSTM model. (train from scratch architecture.)
	'''

	class_weight = {0: 0.05, 1: 0.95}
	training_to_all_data_ratio = 0.5
	nb_cross_val = 1
	nb_epoch = 1000
	batch_size = 32

	video_sequence = dataset.video
	label = dataset.risk_one_hot

	model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
	model.build_cnn_to_lstm_model(input_shape=video_sequence.shape[1:])
	model.train_n_fold_cross_val(video_sequence, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0)
	
	# storing the model weights to cache folder.
	cache_folder = Path('../cache').resolve().mkdir(exist_ok=True)
	model.model.save(str(cache_folder / 'maskRCNN_CNN_lstm_GPU.h5'))
	

def process_raw_images_to_masked_images(src_path: Path, dst_path: Path, coco_path: Path):
    ''' 
        This step is for preprocessing the raw images 
        to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
    '''
    masked_image_extraction = DetectObjects(src_path, dst_path, coco_path)
    masked_image_extraction.save_masked_images()


if __name__ == '__main__':
	
	root_folder_path = Path('../input/synthesis_data').resolve()
	raw_image_path = root_folder_path / 'lane-change'
	label_table_path = root_folder_path / "LCTable.csv"
	
	do_mask_rcnn = True

	if do_mask_rcnn: 
		coco_model_path = Path('../pretrained_models')
		masked_image_path = root_folder_path / (raw_image_path.stem + '_masked') # the path in parallel with raw_image_path
		masked_image_path.mkdir(exist_ok=True)
		process_raw_images_to_masked_images(raw_image_path, masked_image_path, coco_model_path)
		
		#load masked images
		dataset = load_dataset(masked_image_path, label_table_path)

	else:
		#load raw images
		dataset = load_dataset(raw_image_path, label_table_path)
	
	# train import from core
	# output store in maskRCNN_CNN_lstm_GPU.h5
	# use prediction script to evaluate
	train_cnn_to_lstm(dataset)