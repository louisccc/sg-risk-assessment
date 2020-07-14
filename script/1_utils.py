import pandas as pd
import csv
import json
from pathlib import Path
import sys, os,glob
from argparse import ArgumentParser
from tqdm import tqdm
import shutil

class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for writing to LCTable.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data/lane-change", help="Path to input.")
        self.parser.add_argument('--src_path', type=str, default="../input/synthesis_data", help="Path to source.")
        self.parser.add_argument('--dest_path', type=str, default="../input/synthesis_data", help="Path to destination.")
        self.parser.add_argument('--risk_label', type=lambda x: (str(x).lower() == 'true'), default=False, help='Write risk label as txt file')
        self.parser.add_argument('--csv', type=lambda x: (str(x).lower() == 'true'), default=False, help='Create LCtable.csv')

        self.parser.add_argument('--task', type=str, default='createLCtable', help='Task to perform')

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()
        self.src_base_dir = Path(self.src_path).resolve()
        self.dest_base_dir = Path(self.dest_path).resolve()

def create_csv(input_path):
	lctable = input_path / 'LCTable.csv'	

	foldernames = [f for f in sorted(os.listdir(input_path)) if f.isnumeric() and not f.startswith('.')]
	foldernames = sorted(foldernames,key=int)

	column_headers = ['folder_id', 'lc_dir', 'score', 'total_frames', 'img_path', 'json_path', 'gif_path']
	df = pd.DataFrame(columns=column_headers)

	for index, foldername in enumerate(tqdm(foldernames)):
		lc_path = input_path / foldername
		gif_path = lc_path / "lane_change.gif"
		label_path = lc_path / "label.txt"
		json_path = list((lc_path / "scene_raw").glob("*.json"))[0] 
		
		json_data = parse_json(json_path)

		if label_path.exists():
			with open(str(label_path), 'r') as label_f:
				risk_label = int(float(label_f.read().strip().split(",")[0]))
		else:
			raise FileNotFoundError("No label.txt in %s" % label_path) 
		
		images =  list(lc_path.glob("raw_images/*.jpg")) + list(lc_path.glob("raw_images/*.png")) 
		total_frames = len(images)
		df.loc[index] = [foldername, json_data['lc_dir'], risk_label, total_frames, lc_path, json_path, gif_path]
		
	df.to_csv(lctable, encoding='utf-8', index=False)

def parse_json(json_path):
	data_dict = {}
	with open(str(json_path), 'r') as json_file:
		framedict = json.loads(json_file.read())
		frames = list(sorted(framedict.keys(), key=int))
		starting_lane_idx = framedict[frames[-1]]['ego']['orig_lane_idx']
		final_lane_idx = framedict[frames[-1]]['ego']['lane_idx']
		if final_lane_idx < starting_lane_idx:
			data_dict['lc_dir'] = 'left'
		elif final_lane_idx > starting_lane_idx:
			data_dict['lc_dir'] = 'right'
		else:
			data_dict['lc_dir'] = 'fail'
	return data_dict

def copy_csv(file_path):
	# copy risk label from another csv
	input_path = file_path / 'lane-change-100-balanced'
	old_csv = file_path / 'lane-change-804' / 'LCTable.csv'
	new_csv = file_path / 'lane-change-100-balanced' / 'LCTable.csv'

	old_df = pd.read_csv(old_csv, header=None, index_col=None)
	new_df = pd.read_csv(new_csv, header=None, index_col=None)
	foldernames = [f for f in os.listdir(input_path) if f.isnumeric()]
	foldernames = sorted(foldernames,key=int)
	
	for index,foldername in enumerate(tqdm(foldernames)):
		new_df.iloc[index,6] = old_df.iloc[int(foldername),6]

	new_df.to_csv(new_csv,header=None,index=None)

def renumber(file_path):
	# renumber folders to be consecutive for nagoya pipeline
	input_path = file_path / 'lane-change-100-balanced_masked'
	lctable = input_path / 'LCTable.csv'

	df = pd.read_csv(lctable, header=None, index_col=None)
	foldernames = [f for f in sorted(os.listdir(input_path)) if f.isnumeric()]
	foldernames = sorted(foldernames,key=int)
	for index,foldername in enumerate(tqdm(foldernames)):
		df.iloc[index,0] = index
		os.rename(str(input_path)+'/'+str(foldername), str(input_path)+'/'+str(index)) 

	df.to_csv(lctable,header=None,index=None)	

def write_risk_label(file_path):
	input_path = file_path / 'lane-change-804'
	lctable = input_path / 'LCTable.csv'

	df = pd.read_csv(lctable, header=None, index_col=None)
	
	foldernames = [f for f in sorted(os.listdir(input_path)) if f.isnumeric()]
	foldernames = sorted(foldernames,key=int)
	print(foldernames)

	for foldername in tqdm(foldernames):
		fout = open(input_path / foldername / 'label.txt', 'w')
		label = df.iloc[int(foldername),-1]
		fout.write(str(label))
		fout.close()

def merge_dataset(src_path, dest_path):

	foldernames = [f for f in os.listdir(dest_path) if f.isnumeric()]
	foldernames = sorted(foldernames,key=int)

	max_dest_index = int(foldernames[-1])+1 if len(foldernames)>0 else 0

	src_foldernames = [f for f in os.listdir(src_path) if f.isnumeric()]
	src_foldernames = sorted(src_foldernames,key=int)
	#import pdb;pdb.set_trace()

	for index,src_foldername in enumerate(tqdm(src_foldernames)):
		
		new_foldername = str(index+max_dest_index)
		new_path = dest_path / new_foldername
		old_path = src_path / src_foldername

		try:
			shutil.copytree(old_path,new_path)
		except shutil.Error as err:
		    #import pdb;pdb.set_trace()
		    errors = err.args[0]
		    for error in errors:
		        src, dst, msg = error
		        print(src,dst)
		        shutil.copy2(src, dst)

	
def clean(input_path):
	foldernames = [f for f in os.listdir(input_path) if f.isnumeric()]
	foldernames = sorted(foldernames,key=int)

	for index, foldername in enumerate(tqdm(foldernames)):
		carla_visualization = input_path / foldername / "carla_visualize"
		obj_detection_result = input_path / foldername / "obj_det_results"

		shutil.rmtree(carla_visualization)
		shutil.rmtree(obj_detection_result)

if __name__ == '__main__':
	config = Config(sys.argv[1:])

	if config.task=='createLCtable':
		if config.csv == True:
			create_csv(config.input_base_dir)
			
		#write label.txt to video folder
		if config.risk_label == True:
			write_risk_label(config.input_base_dir)

	elif config.task=='merge':
		#merge 2 datasets
		merge_dataset(config.src_base_dir,config.dest_base_dir)
	
	elif config.task=='copy':
		copy_csv(config.input_base_dir)

	elif config.task=='renumber':
		renumber(config.input_base_dir)

	elif config.task=='clean':
		clean(config.input_base_dir)
