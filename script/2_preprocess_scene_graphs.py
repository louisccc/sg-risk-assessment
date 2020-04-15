import os, pdb, sys
sys.path.append(os.path.dirname(sys.path[0]))
import core.graph_learning.utils
import pickle as pkl
from collections import defaultdict


#preprocess data for input to graph learning algorithms.
input_source = "../input/synthesis_data/lane-change-9.8/scenes/"


def get_scene_graphs(dir):
    scenegraphs = defaultdict()
    _, _ , files = os.walk(dir)
    for file in files:
        if ".pkl" in file:
            scenegraphs[file] = pkl.load(file)
    return scenegraphs


if __name__ == "__main__":
    scenegraphs = get_scene_graphs(input_source)
    embeddings = defaultdict()
    for frame_id, scenegraph in scenegraphs:
        embeddings[frame_id] = utils.get_node_embeddings(scenegraph)
    pdb.set_trace()
