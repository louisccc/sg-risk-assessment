import os, pdb, sys
sys.path.append(os.path.dirname(sys.path[0]))
from core.graph_learning import utils
from core.scene_graph import scene_graph, relation_extractor
import pickle as pkl
from collections import defaultdict


#preprocess data for input to graph learning algorithms.
input_source = "input/synthesis_data/lane-change-9.8/scenes/"


def get_scene_graphs(dir):
    scenegraphs = defaultdict()
    root, dir, files = next(os.walk(dir))
    for file in files:
        if ".pkl" in file:
            with open(root + '/' + file, 'rb') as f:
                scenegraphs[file] = pkl.load(f)
            #break #For testing only. returns single graph
    return scenegraphs

#TODO: save extracted embeddings, labels, and adj matrices as pkl files
def save_preprocessed_data(savedir, scenegraphs):
    pass


if __name__ == "__main__":
    scenegraphs = get_scene_graphs(input_source)
    embeddings = defaultdict()
    adj_matrix = defaultdict()
    feature_list = utils.get_feature_list(scenegraphs.values())
    for frame_id, scenegraph in scenegraphs.items():
        embeddings[frame_id] = utils.create_node_embeddings(scenegraph, feature_list)
        adj_matrix[frame_id] = utils.get_adj_matrix(scenegraph)
        #TODO: create onehot labels matrix from ActorType of each entity_node
    pdb.set_trace()
