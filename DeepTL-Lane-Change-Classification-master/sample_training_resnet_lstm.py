from dataset import *
from models import Models
from keras.applications import ResNet50
from keras.models import Model
import os

def preprocess():
	'''
		create backbone model for LSTM
	'''
	backbone_model = ResNet50(weights='imagenet')
	backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.get_layer(index=-2).output)

	return backbone_model

def load_dataset(backbone_model):
	'''
		load the dataset, extract features from video clips using backbone model, 
		classify riskiness of data
	'''
	dir_name = os.path.dirname(__file__)
	dir_name = os.path.dirname(dir_name)

	image_path = os.path.join(dir_name, 'data/input')

	data = DataSet()
	data.model = backbone_model
	data.extract_features(image_path, option='fixed frame amount', number_of_frames=50)
	data.read_risk_data("data/LCTable.csv")
	data.convert_risk_to_one_hot(risk_threshold=0.5)

	return data

def train_resnet_to_lstm(data):
	'''
		build and train lstm model from resnet
	'''
	class_weight = {0: 0.05, 1: 0.95}
	training_to_all_data_ratio = 0.5
	nb_cross_val = 1
	nb_epoch = 1000
	batch_size = 32

	Data = data.video_features
	label = data.risk_one_hot
	
	model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
	model.build_transfer_LSTM_model(input_shape=Data.shape[1:])
	model.train_n_fold_cross_val(Data, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0)
	
	if not os.path.exists('cache/'):
		os.makedirs('cache/')

	model.model.save('cache/new_resnet_lstm.h5')

if __name__ == "__main__":
	backbone_model = preprocess()
	data = load_dataset(backbone_model)
	train_resnet_to_lstm(data)