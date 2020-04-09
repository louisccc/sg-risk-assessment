from dataset import *
from models import Models
import os

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

