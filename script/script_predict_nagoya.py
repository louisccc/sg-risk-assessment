import sys
sys.path.append('../core')

import graph_learning.utils as utils
from nagoya.dataset import *
from keras.models import load_model     
        
class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for creating gifs of input videos.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data", help="Path to data directory.")

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()



if __name__ == '__main__':
	config = Config(sys.argv[1:])

	root_folder_path = config.input_base_dir #Path('../input/synthesis_data').resolve()
	raw_image_path = root_folder_path / 'lane-change-100-old'
	label_table_path = raw_image_path / "LCTable.csv"
	masked_image_path = root_folder_path / (raw_image_path.stem + '_masked') # the path in parallel with raw_image_path
	cache_model_path = Path('../cache').resolve() / 'maskRCNN_CNN_lstm_GPU.h5'

	if not cache_model_path.exists():
		print("Please train the model first.")
		#return -1

	dataset = load_dataset(masked_image_path, label_table_path)
	model   = load_model(str(cache_model_path))
	
	'''
		determine how safe and dangerous each lane change is 
	'''

	true_label = np.argmax(dataset.risk_one_hot,axis=-1)
	output = model.predict_proba(dataset.video)
	metrics = utils.get_scoring_metrics(output,true_label,"risk_classification") 
	print(metrics)
	print(' safe | dangerous \n', model.predict_proba(dataset.video))
	import pdb;pdb.set_trace()
