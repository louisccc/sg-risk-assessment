# av_av
This project is about a collection of approaches to augment perception with scene graphs.

We primarily use [CARLA](https://github.com/carla-simulator/carla) 0.9.8 which is an open-source autonomous car driving simulator. Besides, we also utilized the [scenario_runner](https://github.com/carla-simulator/scenario_runner) which was designed for CARLA challenge event.

For running CARLA on windows 10, we download the official compiled version from CARLA [website](https://github.com/carla-simulator/carla/releases/tag/0.9.8).


We integrated DeepTL-Lane-Change-Classification.
Infers the risk level of lane change video clips with deep learning. Utilizes deep transfer learning (TL) and spatiotemporal networks. https://arxiv.org/abs/1906.02859 


        @INPROCEEDINGS{yurtsever2019,
        author={E. {Yurtsever} and Y. {Liu} and J. {Lambert} and C. {Miyajima} and E. {Takeuchi} and K. {Takeda} and J. H. L. {Hansen}},
        booktitle={2019 IEEE Intelligent Transportation Systems Conference (ITSC)},
        title={Risky Action Recognition in Lane Change Video Clips using Deep Spatiotemporal Networks with Segmentation Mask Transfer},
        year={2019},
        pages={3100-3107},
        doi={10.1109/ITSC.2019.8917362},
        ISSN={null},
        month={Oct},}

> E. Yurtsever, Y. Liu, J. Lambert, C. Miyajima, E. Takeuchi, K. Takeda,and  J.  H.  L.  Hansen,  “Risky  action  recognition  in  lane  change  video clips  using  deep  spatiotemporal  networks  with  segmentation  mask transfer,” in 2019 IEEE Intelligent Transportation Systems Conference (ITSC), Oct 2019, pp. 3100–3107

## Features: 
* A novel deep learning based driving risk assessment framework
* Implemented in Keras
* Using only a monocular camera for the task
* Two versions are available with trained weights! : ResNet50 (TL) + LSTM and MaskRCNN (TL) + CNN + LSTM.
* Two lane change video samples are provided in data/input/lc_samples_50frames.tar.gz 

## Installation:

1- Install the  dependencies:

```shell
	conda install python=3.6.8
	conda install -c anaconda cython=0.29.10
	conda install -c aaronzs tensorflow-gpu
	conda install git
	
	Refer to: https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI to download pycocotools
	
	If you have problems downloading pycocotools refer to: https://stackoverflow.com/questions/14372706/visual-studio-cant-build-due-to-rc-exe
	
	pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

```
    $pip install -r requirements.txt
 
 2- Download MaskRCNN weights* from https://www.dropbox.com/s/n81pagybkj8p5w1/mask_rcnn_coco.h5?dl=0 and move it to /input . 
 
 _*This model and weights were originally obtained from [Mask R-CNN implementation by Matterport](https://github.com/matterport/Mask_RCNN)._
   
## Hardware requirements

The following GPU-enabled devices are supported:

    NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher. See the list of CUDA-enabled GPU cards.

Please check https://www.tensorflow.org/install/gpu for installing the neccessary drivers.
