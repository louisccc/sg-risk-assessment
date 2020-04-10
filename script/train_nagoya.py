import sys
from preprocess import *

sys.path.append('../core')
from dataset import *
from train import train_cnn_to_lstm

def label_risk(data):
	'''
		order videos by risk and find top riskiest
	'''
	data.read_risk_data("../input/LCTable.csv")
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

	
