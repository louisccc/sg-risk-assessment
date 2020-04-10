import sys, os
sys.path.append('../core')
from dataset import *
from models import Models


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

if __name__ == '__main__':
	src = '../input/synthesis_data/lane-change/'
	dest = src + '_masked/'

	#raw to masked image
	raw2masked(src,dest)

	#load masked images
	masked_data = load_masked_dataset(dest)

	#match input to risk label in LCTable 
	data = label_risk(masked_data)

	#train import from core
	#output store in maskRCNN_CNN_lstm_GPU.h5
	#use prediction script to evaluate
	train_cnn_to_lstm(data)

	
