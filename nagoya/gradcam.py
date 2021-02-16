import torch
import argparse
import cv2
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torchvision import models, transforms
from models import *

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)
        print(len(self.gradients))

    def __call__(self, x):
        self.gradients = []

        y = self.model(x)
        y.register_hook(self.save_gradient)

        return y

class ModelOutputs():
    """ Class for making a forward pass, and› getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients
    '''
        Customized for CNN_LSTM without batchnorm
    '''
    def __call__(self, x):
        target_activations = []
        for name, module in self.model.ordered_layers:
            if module == self.feature_module:
                x = F.relu(self.model.TimeDistributed(self.feature_extractor, x))
                target_activations = [x]
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            elif "lstm" in name.lower():
                self.model.train()
                if len(x.shape) < 3:
                    x = x.unsqueeze(0)
                _,(x,_) = module(x)
                self.model.eval()
            elif "l3" in name.lower():
                x = module(x)
            else:
                if 'mp' in name.lower():
                    x = self.model.dropout(self.model.TimeDistributed(module, x))
                else:
                    x = F.relu(self.model.TimeDistributed(module, x))
            
        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, cfg):
        self.cfg = cfg
        self.model = model.to(self.cfg.device)
        self.feature_module = feature_module
        self.model.eval() 

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        input_img = input_img.to(self.cfg.device)

        # grad-feature map, output-layer
        features, output = self.extractor(input_img)
        
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = one_hot.to(self.cfg.device)
        
        # mask class with output and track the gradients
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        
        # NOTE: one_hot is connected to output gradient from torch.sum()
        # for some reason model needs to be in train mode for lstm to backprop
        # solution was to set model from train back to eval
        one_hot.backward(retain_graph=True) 

        # Apply same gradient to each frame
        grads_val = np.array([np.squeeze(gradient.cpu().data.numpy(), 0) for gradient in self.extractor.get_gradients()])

        # dummy gradients mean 0 std 30
        # grads_val = np.random.normal(0, 30, features[-1].shape[1:])

        target = features[-1]   
        target = target.cpu().data.numpy()[0, :] 
        
        # Operate on entire frame
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros((target.shape[0], target.shape[-2], target.shape[-1]), dtype=np.float32)
        cams = []
        for f in range(self.model.frames):
            for i, w in enumerate(weights):
                cam[f] += w * target[f, i]
    
            c = compute_gradmap(cam[f], input_img)
            cams.append(c)
        
        return np.array(cams)

def compute_gradmap(cam, input_img):
    cam = np.maximum(cam, 0) # relu
    cam = cv2.resize(cam, input_img.shape[-2:])
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)