import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from core.dynkg_trainer import *
import pandas as pd 

def parse_attn_weights(node_attns, sequences):

    original_batch = node_attns['original_batch']
    pool_perm = node_attns['pool_perm']
    pool_batch = node_attns['pool_batch']
    pool_score = node_attns['pool_score']
    batch_counter = Counter(original_batch)
    batch_deduct = {0: 0}

    node_attns_list = []
    for idx in range(1, len(sequences)):
        batch_deduct[idx] = batch_deduct[idx-1]+batch_counter[idx-1]

    node_index = {}
    node_dict= {}
    for idx, (p, b, s) in enumerate(zip(pool_perm, pool_batch, pool_score)):
        node_index[idx] = p - batch_deduct[b]
        inv_node_order = {v: k for k, v in sequences[b]['node_order'].items()}
        if b not in node_dict:
            node_dict[b] = []
        node_dict[b].append("%s:%f"%(inv_node_order[node_index[idx]], s))
    node_attns_list.append(node_dict)
    return node_attns_list

def inspect_dynamic_kg(args, iterations=1):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''
    
    trainer = DynKGTrainer(args)
    trainer.load_model()
    # outputs, labels, metric, folder_names = trainer.evaluate()
    outputs_train, labels_train, folder_names_train, acc_loss_train, attns_train, node_attns_train = trainer.inference(trainer.training_data, trainer.training_labels)
    outputs_test, labels_test, folder_names_test, acc_loss_test, attns_test, node_attns_test = trainer.inference(trainer.testing_data, trainer.testing_labels)

    columns = ['safe_level', 'risk_level', 'prediction', 'label', 'folder_name', 'attn_weights', 'node_attns_score']
    inspecting_result_df = pd.DataFrame(columns=columns)

    import pdb; pdb.set_trace()
    node_attns_train_proc = []
    for i in range(len(trainer.training_data)):
        node_attns_train_proc += parse_attn_weights(node_attns_train[i], trainer.training_data[i]['sequence'])

    node_attns_test_proc = []
    for i in range(len(trainer.testing_data)):
        node_attns_test_proc += parse_attn_weights(node_attns_test[i], trainer.testing_data[i]['sequence'])
                
    for output, label, folder_name, attns, node_attns in zip(outputs_train, labels_train, folder_names_train, attns_train, node_attns_train_proc):
        inspecting_result_df = inspecting_result_df.append(
            {"safe_level":output[0],
             "risk_level":output[1],
             "prediction": 1 if output[1] > output[0] else 0,
             "label":label,
             "folder_name":folder_name,
             "attn_weights":{idx:value for idx, value in enumerate(attns)},
             "node_attns_score": node_attns}, ignore_index=True
        )
    
    for output, label, folder_name, attns, node_attns in zip(outputs_test, labels_test, folder_names_test, attns_test, node_attns_test_proc):
        inspecting_result_df = inspecting_result_df.append(
            {"safe_level":output[0],
             "risk_level":output[1],
             "prediction": 1 if output[1] > output[0] else 0,
             "label":label,
             "folder_name":folder_name, 
             "attn_weights":{idx:value for idx, value in enumerate(attns)},
             "node_attns_score": node_attns}, ignore_index=True
        )
    inspecting_result_df.to_csv("inspect.csv", index=False, columns=columns)

    metrics = {}
    metrics['train'] = get_metrics(outputs_train, labels_train)
    metrics['train']['loss'] = acc_loss_train

    metrics['test'] = get_metrics(outputs_test, labels_test)
    metrics['test']['loss'] = acc_loss_test

    print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], metrics['train']['auc'], \
          "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], metrics['test']['auc'])

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    """ the entry of dynkg pipeline training """ 
    inspect_dynamic_kg(sys.argv[1:])