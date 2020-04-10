import sys, os
sys.path.append('../core')
from dataset import *
from models import Models

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

	Data = dataset.video
	label = dataset.risk_one_hot

	model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
	model.build_cnn_to_lstm_model(input_shape=Data.shape[1:])
	model.train_n_fold_cross_val(Data, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0)
	
	if not os.path.exists('../cache/'):
		os.makedirs('../cache/')

	model.model.save('../cache/maskRCNN_CNN_lstm_GPU.h5')


def label_risk(data):
	'''
		order videos by risk and find top riskiest
	'''
	data.read_risk_data("../input/synthesis_data/LCTable.csv")
	data.convert_risk_to_one_hot(risk_threshold=0.5)

	return data

def load_masked_dataset(masked_image_path: Path):
    '''
        This step is for loading the dataset, preprocessing the video clips 
        and neccessary scaling and normalizing. Also it reads and converts the labeling info.
    '''
    data = DataSet()
    data.read_video(masked_image_path, option='fixed frame amount', number_of_frames=5, scaling='scale', scale_x=0.1, scale_y=0.1)
    
    return data

def process_raw_images_to_masked_images(src_path: Path, dst_path: Path, coco_path: Path):
    ''' 
        This step is for preprocessing the raw images 
        to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
    '''
    masked_image_extraction = DetectObjects(src_path, dst_path, coco_path)
    masked_image_extraction.save_masked_images()

if __name__ == '__main__':
	src = Path('../input/synthesis_data/lane-change/').resolve()
	dest = src / '_masked/'
	coco_path = Path('../pretrained_models')

	process_raw_images_to_masked_images(src, dest, coco_path)

	# #load masked images
	masked_data = load_masked_dataset(dest)

	# #match input to risk label in LCTable 
	# data = label_risk(masked_data)

	# #train import from core
	# #output store in maskRCNN_CNN_lstm_GPU.h5
	# #use prediction script to evaluate
	# train_cnn_to_lstm(data)

	
