import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from core.dynkg_trainer import *

def inspect_dynamic_kg(args, iterations=1):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''
    
    trainer = DynKGTrainer(args)
    trainer.load_model()
    outputs, labels, metric, folder_names = trainer.evaluate()

    for output, label, folder_name in zip(outputs, labels, folder_names):
        print(output, label, folder_name)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    """ the entry of dynkg pipeline training """ 
    inspect_dynamic_kg(sys.argv[1:])