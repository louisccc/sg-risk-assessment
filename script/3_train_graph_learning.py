import sys, os, argparse, pdb
sys.path.append(os.path.dirname(sys.path[0]))
from core.graph_learning.node_trainer import GCNTrainer
from core.graph_learning.graph_trainer import GINTrainer
from core.graph_learning.dyngraph_trainer import DynGINTrainer
from core.graph_learning import utils

def get_config(args):
    task_parser = argparse.ArgumentParser()
    task_parser.add_argument('--task', type=str, default="node_classification", help="Task to be executed.")
    config = task_parser.parse_known_args(args)
    return config # <parsed config>, <unknown list of argv>


if __name__ == "__main__":
    config, other_argvs = get_config(sys.argv[1:])

    if config.task == "node_classification":
        # classify the label for each node using the attributes.
        trainer = GCNTrainer(other_argvs)

    elif config.task == "graph_classification":
        # classify graph by duplicating the risk label.
        trainer = GINTrainer(other_argvs)

    elif config.task =="dyngraph_classification":
        # classify a sequence of graphs using the risk label.
        trainer = DynGINTrainer(other_argvs)

    trainer.build_model()
    trainer.train()
    outputs, labels = trainer.predict()
    utils.save_outputs(trainer.config.input_base_dir, outputs, labels, config.task)
    acc = utils.overall_accuracy(outputs, labels)
    print("overall accuracy: " + str(acc))
    pdb.set_trace()