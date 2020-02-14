from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

class GraphVisualizer:

    def __init__(self, img_dir, extension=".jpg"):
        self.img_dir = Path(img_dir).resolve()
        self.img_paths = list(self.img_dir.glob("*" + extension))

    def get_graph(self, img_path):
        #TODO: generate and return scene graph of input img
        return nx.complete_graph(random.randint(10,50))

    def update(self, num, ax_graph, ax_img):
        ax_graph.clear()
        ax_img.clear()

        # update graph
        nx.draw(self.get_graph(self.img_paths[num]), ax=ax_graph)
        img = plt.imread(self.img_paths[num])

        # update image
        ax_img.imshow(img)

        # Set the title
        ax_graph.set_title("Risk {}".format(random.random()))
        ax_img.set_title("Frame {}".format(num))


    def show(self):
        # Build figure
        fig, (ax_graph, ax_img) = plt.subplots(1, 2, figsize=(20, 12))
        fig.canvas.set_window_title("Scene Graph Visualization")
        ani = animation.FuncAnimation(fig, self.update, frames=len(self.img_paths), fargs=(ax_graph, ax_img))
        plt.show()

if __name__ == "__main__":
    g = GraphVisualizer("input/lane-change/")
    g.show()