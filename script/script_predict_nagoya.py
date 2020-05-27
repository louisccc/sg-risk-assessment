import sys
sys.path.append('../core')

from nagoya.dataset import *
from keras.models import load_model


if __name__ == '__main__':
	root_folder_path = Path('../input/synthesis_data').resolve()
	raw_image_path = root_folder_path / 'lane-change'
	label_table_path = raw_image_path / "LCTable.csv"
	masked_image_path = root_folder_path / (raw_image_path.stem + '_masked') # the path in parallel with raw_image_path
	cache_model_path = Path('../cache').resolve() / 'maskRCNN_CNN_lstm_GPU.h5'

	if not cache_model_path.exists():
		print("Please train the model first.")
		return 

	dataset = load_dataset(masked_image_path, label_table_path)
	model   = load_model(str(cache_model_path))
	
	'''
		determine how safe and dangerous each lane change is 
	'''
	print(' safe | dangerous \n', model.predict_proba(dataset.video))
