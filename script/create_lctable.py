import pandas as pd
import csv
from pathlib import Path
import sys, os,glob
from argparse import ArgumentParser
from tqdm import tqdm
import shutil

class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for writing to LCTable.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data", help="Path to input.")
        self.parser.add_argument('--src_path', type=str, default="../input/synthesis_data", help="Path to source.")
        self.parser.add_argument('--dest_path', type=str, default="../input/synthesis_data", help="Path to destination.")
        self.parser.add_argument('--risk_label', type=lambda x: (str(x).lower() == 'true'), default=False, help='Recursive loading gifs')
        self.parser.add_argument('--data_path', type=lambda x: (str(x).lower() == 'true'), default=False, help='Recursive loading gifs')

        self.parser.add_argument('--task', type=str, default='createLCtable', help='Task to perform')

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()
        self.src_base_dir = Path(self.src_path).resolve()
        self.dest_base_dir = Path(self.dest_path).resolve()

def write_data_path(file_path):
	input_path = file_path / 'lane-change'
	lctable = input_path / 'LCTable.csv'	

	df = pd.read_csv(lctable, header=None, index_col=None)
	
	foldernames = [f for f in sorted(os.listdir(input_path)) if f.isnumeric()]
	foldernames = sorted(foldernames,key=int)

	for foldername in tqdm(foldernames):
		video_path = input_path / foldername
		gif_path = video_path / "lane_change.gif"

		df.iloc[int(foldername),0] = int(foldername)

		#video path in column 3, gif path in column 4
		df.iloc[int(foldername),3] = video_path
		df.iloc[int(foldername),4] = gif_path
		
	df.to_csv(lctable,header=None,index=None)

def write_risk_label(file_path):
	input_path = file_path / 'lane-change'
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

		shutil.copytree(old_path,new_path)

	

if __name__ == '__main__':
	config = Config(sys.argv[1:])
	

	if config.task=='createLCtable':
		#write video and gif path to table
		if config.data_path == True:
			write_data_path(config.input_base_dir)	

		#write label.txt to video folder
		if config.risk_label == True:
			write_risk_label(config.input_base_dir)

	elif config.task=='merge':
		#merge 2 datasets
		merge_dataset(config.src_base_dir,config.dest_base_dir)