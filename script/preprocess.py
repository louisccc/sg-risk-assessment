import os, sys

sys.path.append('../core')
from dataset import *
from models import Models
from train import train_cnn_to_lstm
from Mask_RCNN.mask_rcnn.detect_objects import DetectObjects

def raw2masked(image_path, masked_image_path):
	''' 
		This step is for preprocessing the raw images 
		to semantic segmented images (Using Mask RCNN) and store it in [data_path]/masked_images/
	'''
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
	
	return data
