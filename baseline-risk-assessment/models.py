import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LSTM_Classifier(nn.Module):
    '''
    Recurrent Network binary classifier
    Supports 3 models: {GRU, LSTM, LSTM+Dropout}
    
    To call module provide the input_shape, model_name, and cfg params
    input_shape should be a tensor -> (batch_size, frames, channels, height, width) 
    model_name must be one of these {gru, lstm}
        the lstm model can be configured with dropout if cfg.dropout > 0
    '''
    def __init__(self, input_shape, model_name, cfg):
        super(LSTM_Classifier, self).__init__()
        self.cfg = cfg
        self.model_name = model_name
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
        self.dropout = nn.Dropout(self.cfg.dropout)

        if self.model_name == 'gru':
            self.l1 = nn.GRU(input_size=self.channels*self.height*self.width, hidden_size=100, batch_first=True)
            self.l2 = nn.Linear(in_features=100, out_features=2)
        
        elif self.model_name == 'lstm':
            self.l1 = nn.LSTM(input_size=self.channels*self.height*self.width, hidden_size=512, batch_first=True)
            self.l2 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
            self.l3 = nn.Linear(in_features=512, out_features=1000)
            self.l4 = nn.Linear(in_features=1000, out_features=200)
            self.l5 = nn.Linear(in_features=200, out_features=2)

    def forward(self, x):
        # format input for lstm
        # x = self.reshape(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        if self.model_name == 'gru':
            _,l1 = self.l1(x) # return only last sequence
            l2 = self.l2(l1)
            return l2.squeeze()
        elif self.model_name == 'lstm':
            dropout = lambda curr_layer: self.dropout(curr_layer) if self.cfg.dropout != 0 else curr_layer
            l1,_ = self.l1(x)  # return all sequences
            _,(l2,_) = self.l2(l1) # return only last sequence
            l3 = self.l3(dropout(l2))
            l4 = self.l4(dropout(l3))
            l5 = self.l5(l4)
            return l5.squeeze()
        else:
            raise Exception('Unsupported model! Choose between gru or lstm') 

class CNN_LSTM_Classifier(nn.Module):
    '''
    CNN+LSTM binary classifier

    To call module provide the input_shape and cfg params
    input_shape should be a tensor -> (batch_size, frames, channels, height, width) 
    '''
    def __init__(self, input_shape, cfg):
        super(CNN_LSTM_Classifier, self).__init__()
        self.cfg = cfg
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.kernel_size = (3, 3)
        self.lstm_layers = 1
        self.conv_size = lambda i, k, p, s: int((i-k+2*p)/s + 1)
        self.pool_size = lambda i, k, p, s, pool : conv_size(i, k, p, s) // pool + 1
        self.flat_size = lambda f, h, w : f*h*w
        self.TimeDistributed = lambda curr_layer, prev_layer : torch.stack([curr_layer(prev_layer[:,i]) for i in range(self.frames)], dim=1)

        # Note: conv_size and pool_size only work for square 2D matrices, if not a square matrix, run once for height dim and another time for width dim
        '''
        conv_size = lambda i, k, p, s: int((i-k+2*p)/s + 1)
        pool_size = lambda i, k, p, s, pool : conv_size(i, k, p, s) // pool + 1
        flat_size = lambda f, h, w : f*h*w
        '''
        self.bn1 = nn.BatchNorm3d(num_features=5)
        self.bn2 = nn.BatchNorm3d(num_features=5)
        self.bn3 = nn.BatchNorm3d(num_features=5)
        self.bn4 = nn.BatchNorm1d(num_features=5)
        self.bn5 = nn.BatchNorm1d(num_features=5)

        self.c1 = nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=self.kernel_size)
        self.c2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.flat = nn.Flatten(start_dim=1)
        self.flat_dim = self.get_flat_dim()
        self.l1 = nn.Linear(in_features=self.flat_dim, out_features=200)
        self.l2 = nn.Linear(in_features=200, out_features=50)
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=20, num_layers=self.lstm_layers, batch_first=True) 
        self.l3 = nn.Linear(in_features=20, out_features=2)
    
    def get_flat_dim(self):
        c1_h = self.conv_size(self.height, self.kernel_size[-1], 0, 1)
        c1_w = self.conv_size(self.width, self.kernel_size[-1], 0, 1)
        c2_h = self.conv_size(c1_h, self.kernel_size[-1], 0, 1)
        c2_w = self.conv_size(c1_w, self.kernel_size[-1], 0, 1)
        mp1_h = c2_h // 2
        mp1_w = c2_w // 2
        return self.flat_size(16, mp1_h, mp1_w)

    def forward(self, x):
        # Distribute learnable layers across all frames with shared weights
        if self.cfg.bnorm: # can use a larger learning rate w/ bnorm
            c1 = F.relu(self.bn1(self.TimeDistributed(self.c1, x)))
            c2 = F.relu(self.bn2(self.TimeDistributed(self.c2, c1)))
            mp1 = self.dropout(self.bn3(self.TimeDistributed(self.mp1, c2)))
            flat = self.TimeDistributed(self.flat, mp1)
            l1 = F.relu(self.bn4(self.TimeDistributed(self.l1, flat)))
            l2 = F.relu(self.bn5(self.TimeDistributed(self.l2, l1)))
            _,(lstm1,_) = self.lstm1(l2)
            l3 = self.l3(lstm1)
        else:
            c1 = F.relu(self.TimeDistributed(self.c1, x))
            c2 = F.relu(self.TimeDistributed(self.c2, c1))
            mp1 = self.dropout(self.TimeDistributed(self.mp1, c2))
            flat = self.TimeDistributed(self.flat, mp1)
            l1 = F.relu(self.TimeDistributed(self.l1, flat))
            l2 = F.relu(self.TimeDistributed(self.l2, l1))
            _,(lstm1,_) = self.lstm1(l2)
            l3 = self.l3(lstm1)
        
        self.layer_names = self.ordered_layers = [("c1", self.c1),("c2", self.c2),("mp1", self.mp1),("flat", self.flat), ("l1", self.l1),("l2", self.l2),("lstm1", self.lstm1),("l3", self.l3)]
        return l3.squeeze()

class CNN_Classifier(nn.Module):
    '''
    3D CNN+Linear binary classifier

    To call module provide the input_shape and cfg params
    input_shape should be a tensor -> (batch_size, frames, channels, height, width) 
    '''
    def __init__(self, input_shape, cfg):
        super(CNN_Classifier, self).__init__()
        self.cfg = cfg
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
        self.kernel_size = (1, 5, 5)
        self.conv_size = lambda i, k, p, s: int((i-k+2*p)/s + 1)
        self.pool_size = lambda i, k, p, s, pool : conv_size(i, k, p, s) // pool + 1

        self.c1 = nn.Conv3d(in_channels=self.channels, out_channels=32, kernel_size=self.kernel_size)
        self.c2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=self.kernel_size)
        self.mp1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.mp2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.flat = nn.Flatten(start_dim=1)
        self.flat_dim = 64*self.frames*self.get_flat_dim() # TODO: automate this number
        self.l1 = nn.Linear(in_features=self.flat_dim, out_features=1000)
        self.l2 = nn.Linear(in_features=1000, out_features=2)
    
    def get_flat_dim(self):
        c1_h = self.conv_size(self.height, self.kernel_size[-1], 0, 1)
        c1_w = self.conv_size(self.width, self.kernel_size[-1], 0, 1)
        mp1_h = c1_h // 2
        mp1_w = c1_w // 2
        c2_h = self.conv_size(mp1_h, self.kernel_size[-1], 0, 1)
        c2_w = self.conv_size(mp1_w, self.kernel_size[-1], 0, 1)
        mp2_h = c2_h // 2
        mp2_w = c2_w // 2
        return mp2_h * mp2_w

    def reshape(self, x):
        # assumes batch first dim
        return x.permute(0, 2, 1, 3, 4)

    def forward(self, x):
        # format input for 3d cnn
        assert len(x.shape) == 5
        x = self.reshape(x)
        c1 = F.relu(self.c1(x))
        mp1 = self.mp1(c1)
        c2 = F.relu(self.c2(mp1))
        mp2 = self.mp2(c2)
        flat1 = self.flat(mp2)
        l1 = F.relu(self.l1(flat1))
        l2 = self.l2(l1)
        return l2.squeeze()

class ResNet50_LSTM_Classifier(nn.Module):
    '''
    ResNet50+LSTM binary classifier
    
    To call module provide the input_shape, model_name, and cfg params
    input_shape should be a tensor -> (batch_size, frames, channels, height, width) 
    '''
    def __init__(self, input_shape, cfg):
        super(ResNet50_LSTM_Classifier, self).__init__()
        self.cfg = cfg
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
  
        # “Deep Residual Learning for Image Recognition”
        # Using only feature extraction layers shape: (C, H, W) -> (2048, 1, 1)
        '''
        self.resent = models.resnet50(pretrained=True, progress=True)
        nn.Sequential(*list(self.resnet.children())[:-3])(x[0]).shape
        torch.Size([16, 512, 28, 28])
        '''
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True, progress=True).children())[:-1])
        
        # TODO: verify lstm hidden size with louis
        # self.lstm1 = nn.LSTM(input_size=512, hidden_size=20)
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=20, batch_first=True)
        self.l1 = nn.Linear(in_features=20, out_features=2)

    def forward(self, x):
        TimeDistributed = lambda curr_layer, prev_layer : torch.stack([curr_layer(prev_layer[:,i]) for i in range(self.frames)], dim=1)
        resnet = TimeDistributed(self.resnet, x)
        _,(lstm1,_) = self.lstm1(torch.squeeze(resnet))
        l1 = self.l1(lstm1)
        return l1.squeeze()
         
def get_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param)

def test_models():
    # populate some default hyperparameters
    from types import SimpleNamespace
    cfg = {'dropout': 0.1, 'device': 'cpu', 'activation': 'relu'}
    cfg= SimpleNamespace(**cfg)

    # setup fake dataset
    image = torch.randn(16, 5, 3, 224, 224)
    x = image
    
    # construct LSTM_Classifier model
    model_name = ['gru', 'lstm']
    for name in model_name:
        model = LSTM_Classifier(image.shape, name, cfg)
        pred = model(x)
        print(pred.shape, pred)
        assert pred.shape[0] == image.shape[0]
    print('LSTM_Classifier Passed!')

    # construct CNN_LSTM_Classifier model
    model = CNN_LSTM_Classifier(image.shape, cfg)
    pred = model(x)
    print(pred.shape, pred)
    assert pred.shape[0] == image.shape[0]
    print('CNN_LSTM_Classifier Passed!')

    # construct CNN_Classifier model
    model = CNN_Classifier(image.shape, cfg)
    pred = model(x)
    print(pred.shape, pred)
    assert pred.shape[0] == image.shape[0]
    print('CNN_Classifier Passed!')

    # construct ResNet model
    model = ResNet50_LSTM_Classifier(image.shape, cfg)
    pred = model(x)
    print(pred.shape, pred)
    assert pred.shape[0] == image.shape[0]
    print('ResNet_Classifier Passed!')

if __name__ == "__main__":
    test_models()
