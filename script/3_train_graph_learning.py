import os, pdb, sys
sys.path.append(os.path.dirname(sys.path[0]))
from core.graph_learning import utils
from core.graph_learning.models import base_model
import pickle as pkl

#This file trains the graph learning models on scene graph data and saves them in the same input directory as pkl files

input_base_dir = "input/synthesis_data/lane-change-9.8/scenes/"
input_source = input_base_dir + "processed_scenes/"
save_dir = input_base_dir + "models/"

def train_gcn_unsupervised(scenegraphs, embeddings, adj_matrix):
    pass


#loads data from processed_scenes folder. loads labels if they exist.
def load_data(input_source):
    scenegraphs = embeddings = adj_matrix = labels = None
    
    if not os.path.exists(input_source):
        raise FileNotFoundError("path does not exist: " + input_source)
        
    else:
        with open(input_source + "scenegraphs.pkl", 'rb') as f:
            scenegraphs = pkl.load(f)
        with open(input_source + "embeddings.pkl", 'rb') as f:
            embeddings = pkl.load(f)
        with open(input_source + "adj_matrix.pkl", 'rb') as f:
            adj_matrix = pkl.load(f)
        if(os.path.exists(input_source+"labels.pkl")):
            with open(input_source+"labels.pkl", 'rb') as f:
                labels = pkl.load(f)
                
    return scenegraphs, embeddings, adj_matrix, labels


if __name__ == "__main__":
    scenegraphs, embeddings, adj_matrix, labels = load_data(input_source)
    
    #unsupervised learning
    if(labels==None):
        print("Running unsupervised learning")
        train_gcn_unsupervised(scenegraphs, embeddings, adj_matrix)

    #supervised learning
    else:
        raise NotImplementedError("Supervised learning not yet implemented")


