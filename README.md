# av_av
This project is about a collection of approaches that tries to augment autonomous vehicle's perception with scene graphs.

We primarily use [CARLA](https://github.com/carla-simulator/carla) 0.9.8 which is an open-source autonomous car driving simulator. Besides, we also utilized the [scenario_runner](https://github.com/carla-simulator/scenario_runner) which was designed for CARLA challenge event.

For running CARLA on windows 10, we download the official compiled version from CARLA [website](https://github.com/carla-simulator/carla/releases/tag/0.9.8).

In this project,  bnmWe integrated DeepTL-Lane-Change-Classification.
Infers the risk level of lane change video clips with deep learning. Utilizes deep transfer learning (TL) and spatiotemporal networks. https://arxiv.org/abs/1906.02859


# Module Architecture
- **[project folder]/core:** this folder collects all the core functionalities for this project. 
  - Mask_RCNN is the module that handle object detection and coloring on the image sequence. 
  - Nagoya is the module that builts with CNN+LSTM model. 
- **[project folder]/dataset_preparation:** this folder collects modules that help collecting the video clip data regarding lane changes (utilizing [scenario_runner](https://github.com/carla-simulator/scenario_runner) and CARLA 0.9.8
- **[project_folder]/script:** contains the executable scripts that utilize the functions under core. 
- **[project folder]/input:** contains the example inputs for testing purposes. 
- **[project folder]/pretrained_model:** contains the pretrained models or weights required for this project. 
  - Mask-RCNN pretrained weights can refer to [this link](https://www.dropbox.com/s/n81pagybkj8p5w1/mask_rcnn_coco.h5?dl=0) and move it under [project folder]/pretrained_model (credit to [this repo]([Mask R-CNN implementation by Matterport](https://github.com/matterport/Mask_RCNN)). 

# To get started
We recommend our potential users to use [Anaconda](https://www.anaconda.com/) as the primary virtual environment. 

```shell
$conda install python=3.6.8
$conda install -c anaconda cython=0.29.10
$conda install -c aaronzs tensorflow-gpu
$conda install git
$pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
$pip install -r requirements.txt
```	

Since our primary working environment is Windows, we refer to [this solution](https://stackoverflow.com/questions/14372706/visual-studio-cant-build-due-to-rc-exe) to have pycocotools to be installed.

# Usages
To be filled.

# Citation 
Please kindly consider citing our paper if you find our work useful for your research
```
@article{yu2020scene,
  title={Scene-graph augmented data-driven risk assessment of autonomous vehicle decisions},
  author={Yu, Shih-Yuan and Malawade, Arnav V and Muthirayan, Deepan and Khargonekar, Pramod P and Faruque, Mohammad A Al},
  journal={arXiv preprint arXiv:2009.06435},
  year={2020}
}
```
