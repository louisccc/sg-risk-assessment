
from lane_change_risk_detection.models import Models
from lane_change_risk_detection.dataset import DataSet
import os


def preprocess():
	''' 
		This step is for preprocessing the raw images 
		to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
	'''
	from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects

	dir_name = os.path.dirname(__file__)
	dir_name = os.path.dirname(dir_name)

	image_path = os.path.join(dir_name, 'data/input/')
	masked_image_path =os.path.join(dir_name, 'data/masked_images/')

	masked_image_extraction = DetectObjects(image_path, masked_image_path)
	masked_image_extraction.save_masked_images()


def load_dataset():
	'''
		This step is for loading the dataset, preprocessing the video clips 
		and neccessary scaling and normalizing. Also it reads and converts the labeling info.
	'''
	class_weight = {0: 0.05, 1: 0.95}
	training_to_all_data_ratio = 0.9
	nb_cross_val = 1
	nb_epoch = 1000
	batch_size = 32

	data = DataSet()
	data.read_video(masked_image_path, option='fixed frame amount', number_of_frames=50, scaling='scale', scale_x=0.1, scale_y=0.1)

	data.read_risk_data("LCTable.csv")
	data.convert_risk_to_one_hot(risk_threshold=0.05)

	return data


def train_cnn_to_lstm(dataset):
	'''
		This step is for training the CNN to LSTM model. (train from scratch architecture.)
	'''

	Data = dataset.video
	label = dataset.risk_one_hot

	model = Models(nb_epoch=nb_epoch, batch_size=batch_size, class_weights=class_weight)
	model.build_cnn_to_lstm_model(input_shape=Data.shape[1:])
	model.train_n_fold_cross_val(Data, label, training_to_all_data_ratio=training_to_all_data_ratio, n=nb_cross_val, print_option=0, plot_option=0, save_option=0)
	model.model.save('maskRCNN_CNN_lstm_GPU.h5')


if __name__ == "__main__":
	preprocess()
	dataset = load_dataset()
	train_cnn_to_lstm(dataset)
