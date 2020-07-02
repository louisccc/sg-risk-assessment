import sys, pdb, os
sys.path.append(os.path.dirname(sys.path[0]))
from pathlib import Path
from core.scene_graph.scene_graph import SceneGraph, Node
import json

####DEPRECATED####

#used to test that scene graph extraction is working
if __name__ == '__main__':
    # sg = SceneGraph()
    # re = RelationExtractor()
    txt_path = r"..\input\synthesis_data\lane-change\0\scene_raw\25856-25858.txt"
    img_path = r"..\input\synthesis_data\lane-change\0\raw_images"
    store_path = Path(r'.\input\synthesis_data\lane-change\0\scenes')
    store_path.mkdir(parents=True, exist_ok=True)
    framedicts = None
    with open(txt_path, 'rb') as f:
        framedicts = json.load(f)
    
    for key in framedicts.keys():
        sg = SceneGraph(framedicts[key])
        pdb.set_trace()