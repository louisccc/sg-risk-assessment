# Scene-graph Augmented Data-driven Risk Assessment of Autonomous Vehicle Decisions
This repository includes the code and dataset information required for reproducing the results in [our paper](https://arxiv.org/abs/2009.06435). Besides, we also integrated the source code of [our baseline method](https://arxiv.org/abs/1906.02859), [DeepTL-Lane-Change-Classification](https://github.com/Ekim-Yurtsever/DeepTL-Lane-Change-Classification), into this repo. The baseline approach infers the risk level of lane change video clips with deep CNN+LSTM. 

As for fabricating the lane-changing datasets, we use Carla [CARLA](https://github.com/carla-simulator/carla) 0.9.8 which is an open-source autonomous car driving simulator. Besides, we also utilized the [scenario_runner](https://github.com/carla-simulator/scenario_runner) which was designed for CARLA challenge event. For real-driving datasets, we used Honda-Driving Dataset (HDD) in our experiments. We published the converted scene-graph datasets used in our paper [here](http://ieee-dataport.org/3618).

The architecture of this repository is as below:
- **sg-risk-assessment/**: this folder collects all the core model/trainer/utilties used for our scene-graph based approach. 
- **baseline-risk-assessment/**: this folder collects all the related source file that our baseline method requires
  - Mask_RCNN is the module that handle object detection and coloring on the image sequence.
- **sg_risk_assessment.py**: the script that triggers our scene-graph based approach. 
- **baseline_risk_assessment.py**: the script that triggers our baseline algorithm.

# To get started **brandon fix this.
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
To be filled. Brandon fix this.
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
