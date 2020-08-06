import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from core.dynkg_trainer import *
import pandas as pd 

def inspect_dynamic_kg(args, iterations=1):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''
    
    trainer = DynKGTrainer(args)
    trainer.load_model()
    # outputs, labels, metric, folder_names = trainer.evaluate()
    outputs_train, labels_train, folder_names_train, acc_loss_train = trainer.inference(trainer.training_data, trainer.training_labels)
    outputs_test, labels_test, folder_names_test, acc_loss_test = trainer.inference(trainer.testing_data, trainer.testing_labels)

    columns = ['safe_level', 'risk_level', 'prediction', 'label', 'folder_name']
    inspecting_result_df = pd.DataFrame(columns=columns)

    for output, label, folder_name in zip(outputs_train, labels_train, folder_names_train):
        inspecting_result_df = inspecting_result_df.append(
            {"safe_level":output[0],
             "risk_level":output[1],
             "prediction": 1 if output[1] > output[0] else 0,
             "label":label,
             "folder_name":folder_name}, ignore_index=True
        )
    
    for output, label, folder_name in zip(outputs_test, labels_test, folder_names_test):
        inspecting_result_df = inspecting_result_df.append(
            {"safe_level":output[0],
             "risk_level":output[1],
             "prediction": 1 if output[1] > output[0] else 0,
             "label":label,
             "folder_name":folder_name}, ignore_index=True
        )
    inspecting_result_df.to_csv("inspect.csv", index=False, columns=columns)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    """ the entry of dynkg pipeline training """ 
    inspect_dynamic_kg(sys.argv[1:])