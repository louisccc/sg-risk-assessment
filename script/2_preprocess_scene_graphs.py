import os, pdb, sys
sys.path.append(os.path.dirname(sys.path[0]))
from core.graph_learning import utils
from core.scene_graph import scene_graph, relation_extractor
import pickle as pkl
from collections import defaultdict
import numpy as np
import pandas as pd

#preprocess data for input to graph learning algorithms.
input_source = "input/synthesis_data/lane-change-9.8/scenes/"
save_dir = input_source + "processed_scenes/"

def get_scene_graphs(dir):
    scenegraphs = defaultdict()
    root, dir, files = next(os.walk(dir))
    for file in files:
        if ".pkl" in file:
            with open(root + '/' + file, 'rb') as f:
                scenegraphs[file] = pkl.load(f)
            #break #For testing only. returns single graph
    return scenegraphs

#save extracted embeddings, labels, and adj matrices as pkl files
#the entire dictionary for all frames is saved in each pkl file
def save_preprocessed_data(savedir, scenegraphs, embeddings, adj_matrix, labels):
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    
    for frame_id, scenegraph in scenegraphs.items():
        with open(savedir+'scenegraphs.pkl','wb') as f:
            pkl.dump(scenegraphs, f)
        with open(savedir+'embeddings.pkl', 'wb') as f:
            pkl.dump(embeddings, f)
            combined_embeddings = pd.DataFrame()
            for item in embeddings.values():
                combined_embeddings = pd.concat([combined_embeddings, item], axis=0, ignore_index=False)
            combined_embeddings.to_csv(savedir+'combined_embeddings.tsv', sep='\t', header=False, index=False)
        with open(savedir+'adj_matrix.pkl', 'wb') as f:
            pkl.dump(adj_matrix, f)
        if(labels != None):
            with open(savedir+'labels.pkl', 'wb') as f:
                pkl.dump(labels, f)
            pd.DataFrame(np.concatenate(list(labels.values()))).to_csv(savedir+'meta.tsv', sep='\t', header=False, index=False)
            
    print("processed data saved to: " + savedir)

if __name__ == "__main__":
    scenegraphs = get_scene_graphs(input_source)
    embeddings = defaultdict()
    adj_matrix = defaultdict()
    labels = defaultdict()
    feature_list = utils.get_feature_list(scenegraphs.values(), num_classes=8)
    for frame_id, scenegraph in scenegraphs.items():
        labels[frame_id], embeddings[frame_id] = utils.create_node_embeddings(scenegraph, feature_list)
        adj_matrix[frame_id] = utils.get_adj_matrix(scenegraph)
    save_preprocessed_data(save_dir, scenegraphs, embeddings, adj_matrix, labels)
    pdb.set_trace()
