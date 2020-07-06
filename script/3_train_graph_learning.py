import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from core.dynkg_trainer import *
import pandas as pd


def train_dynamic_kg(args, iterations=1):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''
    
    outputs = []
    labels = []
    metrics = []

    for i in range(iterations):
        trainer = DynKGTrainer(args)
        trainer.build_model()
        trainer.train()
        output, label, metric = trainer.evaluate()

        outputs += output
        labels  += label
        metrics.append(metric)

    # Store the prediction results. 
    store_path = trainer.config.input_base_dir
    outputs_pd = pd.DataFrame(outputs)
    labels_pd  = pd.DataFrame(labels)
    
    labels_pd.to_csv(store_path / "dynkg_training_labels.tsv", sep='\t', header=False, index=False)
    outputs_pd.to_csv(store_path / "dynkg_training_outputs.tsv", sep="\t", header=False, index=False)
    
    # Store the metric results. 
    metrics_pd = pd.DataFrame(metrics[-1]['test'], index=[0])
    metrics_pd.to_csv(store_path / "dynkg_classification_metrics.csv", header=True)


if __name__ == "__main__":
    """ the entry of dynkg pipeline training """ 
    train_dynamic_kg(sys.argv[1:])