
import networkx as nx
from relation_extractor import Relations, RelationExtractor

#class defining scene graph and its attributes. contains functions for construction and operations
class SceneGraph:
    
    def __init__(self):
        self.g = nx.Graph() #initialize scenegraph as networkx graph
        
    #add single node to graph
    def add_node(self, node):
        self.g.add_node(node)
        
    #add multiple nodes to graph
    def add_nodes(self, nodes):
        self.g.add_nodes_from(nodes)
    
    #add relation (edge) between nodes on graph
    def add_relation(self, relation):
        self.g.add_edge(relation[0], relation[2], object=relation[1])
        
    def add_relations(self, relations_list):
        for relation in relations_list:
            self.add_relation(relation)
            
            
if __name__ == '__main__':
#demo code
    sg = SceneGraph()
    sg.add_node('a')
    sg.add_node('b')
    sg.add_relation(['a',Relations.isIn, 'b'])
    
    