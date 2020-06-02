import sys, os, argparse, pdb
sys.path.append(os.path.dirname(sys.path[0]))
from core.graph_learning.node_trainer import GCNTrainer
from core.graph_learning.graph_trainer import GraphTrainer
from core.graph_learning.dyngraph_trainer import DynGraphTrainer
from core.graph_learning import utils
import pandas as pd
import torch
import numpy as np

def get_config(args):
    task_parser = argparse.ArgumentParser()
    task_parser.add_argument('--task', type=str, default="node_classification", help="Task to be executed.")
    task_parser.add_argument('--iterations', type=int, default=3, help="Number of times the model should be built and scored for averaging.")
    config = task_parser.parse_known_args(args)
    return config # <parsed config>, <unknown list of argv>
    
def get_trainer(config):
    if config.task == "node_classification":
        # classify the label for each node using the attributes.
        trainer = GCNTrainer(other_argvs)

    elif config.task == "graph_classification":
        # classify graph by duplicating the risk label.
        trainer = GraphTrainer(other_argvs)

    elif config.task =="dyngraph_classification":
        # classify a sequence of graphs using the risk label.
        trainer = DynGraphTrainer(other_argvs)
    return trainer

if __name__ == "__main__":
    config, other_argvs = get_config(sys.argv[1:])
    outputs = None
    labels = None
    trainer = None
    for i in range(config.iterations):
        trainer = get_trainer(config)
        trainer.build_model()
        trainer.train()
        if i == 0:
            outputs, labels = trainer.predict()
        else:
            o, l = trainer.predict()
            outputs = torch.cat((outputs, o))
            labels = np.concatenate((labels, l))
            
    utils.save_outputs(trainer.config.input_base_dir, outputs, labels, config.task)
    #if config.task != "node_classification": #multiclass metrics not implemented
    metrics = utils.get_scoring_metrics(outputs, labels, config.task)
    pd.DataFrame(metrics, index=[0]).to_csv(str(trainer.config.input_base_dir) + "/" + config.task + "_metrics.csv", header=True)
    pdb.set_trace()