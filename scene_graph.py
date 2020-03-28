import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from relation_extractor import Relations, RelationExtractor
import pdb, json, random
from pathlib import Path
from glob import glob
#basic class for abstracting a node in the scene graph. this is mainly used for holding the data for each node.


class Node:
    def __init__(self, name, attr, is_entity=False):
        self.name = name
        self.attr = attr
        self.label = name + "\n" + str(attr)
        self.is_entity = is_entity

    def __repr__(self):
        return "%s" % self.name 
    

#class defining scene graph and its attributes. contains functions for construction and operations
class SceneGraph:
    
    #graph can be initialized with a framedict to load all objects at once
    def __init__(self, framedict=None):
        self.g = nx.Graph() #initialize scenegraph as networkx graph
        self.relation_extractor = RelationExtractor()
        self.road_node = Node("road", None, False)
        self.add_node(self.road_node)   #adding the road as the root node
        self.entity_nodes = []  #nodes which are explicit entities and not attributes.
        
        if framedict != None:
            self.add_frame_dict(framedict)
        
    #add single node to graph. node can be any hashable datatype including objects.
    def add_node(self, node):
        self.g.add_node(node, attr=node.attr, label=node.label)
        if(node.is_entity):
            self.entity_nodes.append(node)
        
    #add multiple nodes to graph
    def add_nodes(self, nodes):
        self.g.add_nodes_from(nodes)
    
    #add relation (edge) between nodes on graph. relation is a list containing [subject, relation, object]
    def add_relation(self, relation):
        if relation != []:
            if relation[0] not in self.g.nodes:
                self.add_node(relation[0])
            if relation[2] not in self.g.nodes:
                self.add_node(relation[2])
            self.g.add_edge(relation[0], relation[2], object=relation[1])
        
    def add_relations(self, relations_list):
        for relation in relations_list:
            self.add_relation(relation)
            
    #parses actor dict and adds nodes to graph. this can be used for all actor types.
    def add_actor_dict(self, actordict):
        for actor_id, attr in actordict.items():
            n = Node(actor_id, attr, True)   #using the actor key as the node name and the dict as its attributes.
            self.add_node(n)
            self.add_attributes(n, attr)
            
    #adds lanes and their dicts. constructs relation between each lane and the root road node.
    def add_lane_dict(self, lanedict):
        for lane_id, laneattr in lanedict.items():
            n = Node(lane_id, laneattr, False) #todo: change to true when lanedict entry is complete
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])
            
    #add signs as entities of the road.
    def add_sign_dict(self, signdict):
        for sign_id, signattr in signdict.items():
            n = Node(sign_id, signattr, True)
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])

    #parses attributes of ego/actors
    def add_attributes(self, node, attrdict):
        for attr, values in attrdict.items():
            n = Node(attr, values, False)   #attribute nodes are not entities themselves
            self.add_node(n)
            self.add_relation([node, Relations.hasAttribute, n])

    #add the contents of a whole framedict to the graph
    def add_frame_dict(self, framedict):
        for key, attrs in framedict.items():
            if key == "ego":
                egoNode = Node(key, attrs, True)
                self.add_node(egoNode)
                self.add_attributes(egoNode, attrs)
            elif key == "lane":
                self.add_lane_dict(attrs)
            elif key == "sign":
                self.add_sign_dict(attrs)
            else:
                self.add_actor_dict(attrs)
            
    #calls RelationExtractor to build semantic relations between every pair of entity nodes in graph. call this function after all nodes have been added to graph.
    def extract_semantic_relations(self):
        for node1 in self.entity_nodes:
            for node2 in self.entity_nodes:
                if node1.name != node2.name: #dont build self-relations
                    self.add_relations(self.relation_extractor.extract_relations(node1, node2))
    

class SceneGraphExtractor:
    def __init__(self):
        self.scenegraphs = {}
        self.scene_images = {}

        # For Visualization
        self.fig, (self.ax_graph, self.ax_img) = plt.subplots(1, 2, figsize=(20, 12))
        self.fig.canvas.set_window_title("Scene Graph Visualization")

    def load(self, path):
        for txt_path in glob("%s/**/*.txt" % str(path), recursive=True):
            with open(txt_path, 'r') as f:
                framedict = json.loads(f.read())
                for frame, frame_dict in framedict.items():
                    scenegraph = SceneGraph()
                    scenegraph.add_frame_dict(frame_dict)
                    scenegraph.extract_semantic_relations()
                    self.scenegraphs[frame] = scenegraph

    def build_corresponding_images(self, path):
        for frame, scenegraph in self.scenegraphs.items():
            try:
                print('catch %s/%.8d.png'%(path, int(frame)))
                img = plt.imread('%s/%.8d.png'%(path, int(frame)))
                self.scene_images[frame] = (scenegraph, img)
            except Exception as e:
                print(e)
        
    def store(self, path):
        for frame, (scenegraph, image) in self.scene_images.items():
            pos = nx.spring_layout(scenegraph.g, k=1.5*1/np.sqrt(len(scenegraph.g.nodes())))
            nx.draw(scenegraph.g, labels=nx.get_node_attributes(scenegraph.g, 'label'), pos=pos, font_size=8, with_labels=True, ax=self.ax_graph)
            self.ax_img.imshow(image)
            self.ax_graph.set_title("Risk {}".format(random.random()))
            self.ax_img.set_title("Frame {}".format(frame))
            plt.savefig('%s/%s.png'%(path, frame))
            self.ax_graph.clear()
            self.ax_img.clear()

    def update(self, num):
        self.ax_graph.clear()
        self.ax_img.clear()

        frame = list(self.scene_images.keys())[num]
        scenegraph = self.scene_images[frame][0]
        pos = nx.spring_layout(scenegraph.g, k=1.5*1/np.sqrt(len(scenegraph.g.nodes())))
        nx.draw(scenegraph.g, labels=nx.get_node_attributes(scenegraph.g, 'label'), pos=pos, font_size=8, with_labels=True, ax=self.ax_graph)
        self.ax_img.imshow(self.scene_images[frame][1])

        # Set the title
        self.ax_graph.set_title("Risk {}".format(random.random()))
        self.ax_img.set_title("Frame {}".format(num))

    def show_animation(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.scene_images.keys()))
        plt.show()

if __name__ == '__main__':
    # sg = SceneGraph()
    # re = RelationExtractor()
    txt_path = r".\input\lane-change-9.8\scene_raw"
    img_path = r".\input\lane-change-9.8\raw_images"
    store_path = Path(r'.\input\lane-change-9.8\scenes')
    store_path.mkdir(parents=True, exist_ok=True)
    
    sge = SceneGraphExtractor()
    sge.load(txt_path)
    sge.build_corresponding_images(img_path)
    sge.store(store_path)
    sge.show_animation()
    pdb.set_trace()
