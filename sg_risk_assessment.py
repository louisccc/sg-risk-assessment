import sys, os
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import wandb

from sg_risk_assessment.dynkg_trainer import *

PROJECT_NAME = "Fill me with wandb id"

class Config:
    '''Argument Parser for script to train scenegraphs.'''
    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for training the scene graph using GCN.')
        self.parser.add_argument('--cache_path', type=str, default="../script/image_dataset.pkl", help="Path to the cache file.")
        self.parser.add_argument('--transfer_path', type=str, default="", help="Path to the transfer file.")
        self.parser.add_argument('--model_load_path', type=str, default="./model/model_best_val_loss_.vec.pt", help="Path to load cached model file.")
        self.parser.add_argument('--model_save_path', type=str, default="./model/model_best_val_loss_.vec.pt", help="Path to save model file.")
        
        # Model
        self.parser.add_argument('--model', type=str, default="mrgcn", help="Model to be used intrinsically. options: [mrgcn, mrgin]")
        self.parser.add_argument('--conv_type', type=str, default="FastRGCNConv", help="type of RGCNConv to use [RGCNConv, FastRGCNConv].")
        self.parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in the network.")
        self.parser.add_argument('--hidden_dim', type=int, default=32, help="Hidden dimension in RGCN.")
        self.parser.add_argument('--layer_spec', type=str, default=None, help="manually specify the size of each layer in format l1,l2,l3 (no spaces).")
        self.parser.add_argument('--pooling_type', type=str, default="sagpool", help="Graph pooling type, options: [sagpool, topk, None].")
        self.parser.add_argument('--pooling_ratio', type=float, default=0.5, help="Graph pooling ratio.")        
        self.parser.add_argument('--readout_type', type=str, default="mean", help="Readout type, options: [max, mean, add].")
        self.parser.add_argument('--temporal_type', type=str, default="lstm_attn", help="Temporal type, options: [mean, lstm_last, lstm_sum, lstm_attn].")
        self.parser.add_argument('--lstm_input_dim', type=int, default=50, help="LSTM input dimensions.")
        self.parser.add_argument('--lstm_output_dim', type=int, default=20, help="LSTM output dimensions.")
        
        # Training
        self.parser.add_argument('--device', type=str, default="cuda", help='The device on which models are run, options: [cuda, cpu].')
        self.parser.add_argument('--downsample', type=lambda x: (str(x).lower() == 'true'), default=False, help='Set to true to downsample dataset.')
        self.parser.add_argument('--nclass', type=int, default=2, help="The number of classes for dynamic graph classification (currently only supports 2).")
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed.')
        self.parser.add_argument('--split_ratio', type=float, default=0.3, help="Ratio of dataset withheld for testing.")
        self.parser.add_argument('--stats_path', type=str, default="best_stats.csv", help="path to save best test statistics.")
        self.parser.add_argument('--test_step', type=int, default=10, help='Number of training epochs before testing the model.')
        
        # Hyperparameters
        self.parser.add_argument('--activation', type=str, default='relu', help='Activation function to use, options: [relu, leaky_relu].')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Number of graphs in a batch.')
        self.parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='The initial learning rate for GCN.')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    

        self.args = args
        args_parsed = self.parser.parse_args(args)
        self.wandb = wandb.init(project=PROJECT_NAME)
        self.wandb_config = self.wandb.config
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
            self.wandb_config[arg_name] = getattr(args_parsed, arg_name)
            
        self.cache_path = Path(self.cache_path).resolve()
        if self.transfer_path != "":
            self.transfer_path = Path(self.transfer_path).resolve()
        else:
            self.transfer_path = None
        self.stats_path = Path(self.stats_path.strip()).resolve()

def train_dynamic_kg(config, iterations=1):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''
    
    outputs = []
    labels = []
    metrics = []

    for i in range(iterations):
        trainer = DynKGTrainer(config)
        trainer.init_dataset()
        trainer.build_model()
        trainer.train()
        categories_train, categories_test, metric, folder_names_train = trainer.evaluate()

        outputs += categories_train['all']['outputs']
        labels  += categories_train['all']['labels']
        metrics.append(metric)

    # Store the prediction results. 
    store_path = trainer.config.cache_path.parent
    outputs_pd = pd.DataFrame(outputs)
    labels_pd  = pd.DataFrame(labels)
    
    labels_pd.to_csv(store_path / "dynkg_training_labels.tsv", sep='\t', header=False, index=False)
    outputs_pd.to_csv(store_path / "dynkg_training_outputs.tsv", sep="\t", header=False, index=False)
    
    # Store the metric results. 
    metrics_pd = pd.DataFrame(metrics[-1]['test'])
    metrics_pd.to_csv(store_path / "dynkg_classification_metrics.csv", header=True)


if __name__ == "__main__":
    """ the entry of dynkg pipeline training """ 
    config = Config(sys.argv[1:])
    train_dynamic_kg(config)