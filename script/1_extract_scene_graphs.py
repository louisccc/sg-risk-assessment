import sys, pdb, os
sys.path.append(os.path.dirname(sys.path[0]))
from pathlib import Path
from core.scene_graph.scene_graph import SceneGraphExtractor, SceneGraph, Node

if __name__ == '__main__':
    # sg = SceneGraph()
    # re = RelationExtractor()
    txt_path = r"..\input\synthesis_data\lane-change\0\scene_raw"
    img_path = r"..\input\synthesis_data\lane-change\0\raw_images"
    store_path = Path(r'.\input\synthesis_data\lane-change\0\scenes')
    store_path.mkdir(parents=True, exist_ok=True)
    
    sge = SceneGraphExtractor()
    sge.load(txt_path)
    # sge.build_corresponding_images(img_path)
    # sge.store(store_path)
    # sge.show_animation()
    pdb.set_trace()