from keras.applications import ResNet50
from keras.models import Model
import os, sys
import argparse

sys.path.append('../core')
from dataset import *
from models import Models
from train import train_cnn_to_lstm

def raw2masked(image_path, masked_image_path):
	''' 
		This step is for preprocessing the raw images 
		to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
	'''
	from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects

	dir_name = os.path.dirname(__file__)
	dir_name = os.path.dirname(dir_name)

	masked_image_extraction = DetectObjects(image_path, masked_image_path)
	masked_image_extraction.save_masked_images()

def load_masked_dataset(masked_image_path):
	'''
		This step is for loading the dataset, preprocessing the video clips 
		and neccessary scaling and normalizing. Also it reads and converts the labeling info.
	'''
	dir_name = os.path.dirname(__file__)
	dir_name = os.path.dirname(dir_name)

	data = DataSet()
	data.read_video(masked_image_path, option='fixed frame amount', number_of_frames=5, scaling='scale', scale_x=0.1, scale_y=0.1)
	data.read_risk_data("../input/LCTable.csv")
	data.convert_risk_to_one_hot(risk_threshold=0.5)

	return data

if __name__ == '__main__':
	src = '../input/synthesis_data/lane-change/'
	dest = src + '_masked/'

	#raw to masked image
	raw2masked(src,dest)

	#load masked images
	data = load_masked_dataset(dest)

	#train import from core
	#output store in maskRCNN_CNN_lstm_GPU.h5
	#use prediction script to evaluate
	train_cnn_to_lstm(data)

	
