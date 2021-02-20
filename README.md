# Scene-graph Augmented Data-driven Risk Assessment of Autonomous Vehicle Decisions
This repository includes the code and dataset information required for reproducing the results in [our paper](https://arxiv.org/abs/2009.06435). Besides, we also integrated the source code of [our baseline method](https://arxiv.org/abs/1906.02859), [DeepTL-Lane-Change-Classification](https://github.com/Ekim-Yurtsever/DeepTL-Lane-Change-Classification), into this repo. The baseline approach infers the risk level of lane change video clips with deep CNN+LSTM. The architecture of our approach is illustrated as below,

![](https://github.com/louisccc/sg-risk-assessment/blob/master/assets/archi.png?raw=true)

As for fabricating the lane-changing datasets, we use Carla [CARLA](https://github.com/carla-simulator/carla) 0.9.8 which is an open-source autonomous car driving simulator. Besides, we also utilized the [scenario_runner](https://github.com/carla-simulator/scenario_runner) which was designed for CARLA challenge event. For real-driving datasets, we used Honda-Driving Dataset (HDD) in our experiments. We published the converted scene-graph datasets used in our paper [here](http://ieee-dataport.org/3618).

The architecture of this repository is as below:
- **sg-risk-assessment/**: this folder consists of all the related source files used for our scene-graph based approach. 
- **baseline-risk-assessment/**: this folder consists of all the related source files used for the baseline method.
- **sg_risk_assessment.py**: the script that triggers our scene-graph based approach. 
- **baseline_risk_assessment.py**: the script that triggers the baseline model.

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
  $ python sg_risk_assessment.py --load_pkl True --pkl_path risk-assessment/scenegraph/synthetic/271_dataset.pkl

  # For tuning hyperparameters view the config class of sg_risk_assessment.py
```

For running the baseline-risk-assessment in this repo, you may refer to the following commands:
```shell
  $ python baseline_risk_assessment.py --load_pkl True --pkl_path risk-assessment/scene/synthetic/271_dataset.pkl

  # For tuning hyperparameters view the config class of baseline_risk_assessment.py
```

After running these commands, the expected outputs are a dump of metrics logged by wandb:
```shell
wandb:                    train_recall ▁████████████████████
wandb:                   val_precision █▁▅▄▅▄▆▆▆▅▄▄▇▆▅▆▅▇▆▆▆
wandb:                      val_recall ▁████████████████████
wandb:                       train_fpr ▁█▅▅▄▅▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂
wandb:                       train_tnr █▁▄▅▅▅▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇
wandb:                       train_fnr █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                         val_fpr ▁█▄▅▄▅▃▃▃▄▄▅▂▃▃▃▄▂▃▃▃
wandb:                         val_tnr █▁▆▄▆▄▆▆▆▆▅▄▇▆▆▆▆▇▆▆▆
wandb:                         val_fnr █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      best_epoch ▁▁▂▂▂▂▃▃▄▄▄▄▅▅▅▅▅▇▇▇█
wandb:                   best_val_loss █▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                    best_val_acc ▁▆█▇█████████████████
wandb:                    best_val_auc ▁▅▆▆▇▇▇▇████▇▇▇▇▇████
wandb:                    best_val_mcc ▁▇███████████████████
wandb:           best_val_acc_balanced ▁████████████████████
wandb:                       train_mcc ▁▇▇▇▇▇███████████████
wandb:                         val_mcc ▁▇███████████████████
```

A graphical visualization of the model outputs including loss and additional metrics can be viewed by creating and linking your runs to [wandb](https://wandb.ai/home).

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
