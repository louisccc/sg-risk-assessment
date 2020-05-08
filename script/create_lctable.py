import pandas as pd
import csv
from pathlib import Path
import sys, os,glob
from argparse import ArgumentParser
from tqdm import tqdm

class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for writing to LCTable.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data", help="Path to input.")

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()

def write_data_path(file_path):
	lctable = file_path / 'LCTable.csv'
	input_path = file_path / 'lane-change'

	df = pd.read_csv(lctable, header=None, index_col=None)
	
	foldernames = [f for f in sorted(os.listdir(input_path)) if f.isnumeric()]
	foldernames = sorted(foldernames,key=int)

	for foldername in tqdm(foldernames):
		video_path = input_path / foldername
		gif_path = video_path / "lane_change.gif"

		#video path in column 3, gif path in column 4
		df.iloc[int(foldername),3] = video_path
		df.iloc[int(foldername),4] = gif_path
		
	df.to_csv(lctable,header=None,index=None)

def write_risk_label(file_path):
	lctable = file_path / 'LCTable.csv'
	input_path = file_path / 'lane-change'

	df = pd.read_csv(lctable, header=None, index_col=None)
	
	foldernames = [f for f in sorted(os.listdir(input_path)) if f.isnumeric()]
	foldernames = sorted(foldernames,key=int)
	print(foldernames)

	for foldername in tqdm(foldernames):
		fout = open(input_path / foldername / 'label.txt', 'w')
		label = df.iloc[int(foldername),-1]
		fout.write(str(label))
		fout.close()


if __name__ == '__main__':
	config = Config(sys.argv[1:])
	write_risk_label(config.input_base_dir)
	#write_data_path(config.input_base_dir)