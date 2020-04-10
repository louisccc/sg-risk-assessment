import sys
sys.path.append('../core')

from dataset import *
from keras.models import load_model

def predict(data):
	'''
		determine how safe and dangerous each lane change is 
	'''
	model = load_model('../cache/maskRCNN_CNN_lstm_GPU.h5')
	print(' safe | dangerous \n', model.predict_proba(data.video))


if __name__ == '__main__':
	src = '../input/synthesis_data/lane-change/'
	dest = src + '_masked/'

	#raw2masked(src, dest)
	data = load_masked_dataset(dest)
	predict(data)


