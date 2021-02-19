# Scene-graph Augmented Data-driven Risk Assessment of Autonomous Vehicle Decisions
This repository includes the code and dataset information required for reproducing the results in [our paper](https://arxiv.org/abs/2009.06435). Besides, we also integrated the source code of [our baseline method](https://arxiv.org/abs/1906.02859), [DeepTL-Lane-Change-Classification](https://github.com/Ekim-Yurtsever/DeepTL-Lane-Change-Classification), into this repo. The baseline approach infers the risk level of lane change video clips with deep CNN+LSTM. The architecture of our approach is illustrated as below,

![](https://github.com/louisccc/sg-risk-assessment/blob/master/archi.png?raw=true)

As for fabricating the lane-changing datasets, we use Carla [CARLA](https://github.com/carla-simulator/carla) 0.9.8 which is an open-source autonomous car driving simulator. Besides, we also utilized the [scenario_runner](https://github.com/carla-simulator/scenario_runner) which was designed for CARLA challenge event. For real-driving datasets, we used Honda-Driving Dataset (HDD) in our experiments. We published the converted scene-graph datasets used in our paper [here](http://ieee-dataport.org/3618).

The architecture of this repository is as below:
- **sg-risk-assessment/**: this folder collects all the core model/trainer/utilties used for our scene-graph based approach. 
- **baseline-risk-assessment/**: this folder collects all the related source file that our baseline method requires
  - Mask_RCNN is the module that handle object detection and coloring on the image sequence.
- **sg_risk_assessment.py**: the script that triggers our scene-graph based approach. 
- **baseline_risk_assessment.py**: the script that triggers our baseline algorithm.

# To Get Started
We recommend our potential users to use [Anaconda](https://www.anaconda.com/) as the primary virtual environment. The requirements to run through our repo are as follows,
- python >= 3.6 
- torch == 1.6.0
- torch_teometric == 1.6.1

Our recommended command sequence is as follows:
```shell
# conda create --name sg_risk_assessment python=3.6
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
$ python -m pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
$ python -m pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
$ python -m pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
$ python -m pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
$ python -m pip install torch-geometric==1.6.1
$ python -m pip install -r requirements.txt
```	
This set of commands assumes you to have cuda10.1 in your local. Please refer to the installation guides of [torch](https://pytorch.org/) and [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) if you have different environment settings.

# Usages
For running the sg-risk-assessment in this repo, you may refer to the following commands:
```shell
  # pre-step, get dataset?
  $ python sg_risk_assessment.py # how to run sg_risk_assessment.py
```

For running the baseline-risk-assessment in this repo, you may refer to the following commands:
```shell
  # pre-step, get dataset?
  $ python baseline_risk_assessment.py # how to run baseline_risk_assessment.py
```

After running these commands the expected outputs are like:
```shell
To be filled.
```

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
