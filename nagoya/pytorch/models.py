from __future__ import print_function
import keras.backend as K
import tensorflow as tf
import torchvision.models as models
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.utils import resample
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import random
import numpy as np
from core.dynkg_trainer import get_metrics
import sys
sys.path.append('../')


# convert keras and tf to pytorch


class Models(nn.Module):

    def __init__(self, nb_epoch=10, batch_size=64, name='default model', class_weights={0: 0.05, 1: 0.95}):
        super(Models, self).__init__()
        self.model = []
        self.weights = []
        self.model_json = []
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.name = name
        self.class_weights = class_weights
        self.history = []
        self.last_Mpercent_epoch_val_loss = []
        self.m_fold_cross_val_results = []
        self.flops = []
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_loss_history(self, X_train, y_train, X_test, y_test):

        class LossHistory(keras.callbacks.Callback):

            def __init__(self, train_x, train_y, val_x, val_y):
                self.train_x = train_x
                self.train_y = train_y
                self.val_x = val_x
                self.val_y = val_y
                self.train_true_label = np.argmax(train_y, axis=-1)
                self.test_true_label = np.argmax(val_y, axis=-1)

                self.AUC_train = []
                self.AUC_val = []
                self.train_loss = []
                self.val_loss = []
                self.f1_train = []
                self.f1_val = []

                self.min_val_loss = float('inf')
                self.best_metrics = {}

            def make_prediction(self, scores, threshold=0.3):
                return [1 if score >= threshold else 0 for score in scores]

            def on_train_begin(self, logs=None):
                y_pred_train = self.model.predict_proba(self.train_x)
                y_pred_val = self.model.predict_proba(self.val_x)

                self.AUC_train.append(
                    roc_auc_score(self.train_y, y_pred_train))
                self.AUC_val.append(roc_auc_score(self.val_y, y_pred_val))

                self.f1_train.append(
                    f1_score(self.train_y[:, 1], self.make_prediction(y_pred_train[:, 1])))
                self.f1_val.append(
                    f1_score(self.val_y[:, 1], self.make_prediction(y_pred_val[:, 1])))

            def on_epoch_end(self, epoch, logs={}):
                y_pred_train = self.model.predict_proba(self.train_x)
                y_pred_val = self.model.predict_proba(self.val_x)

                train_loss = logs.get('loss')
                val_loss = logs.get('val_loss')

                self.AUC_train.append(
                    roc_auc_score(self.train_y, y_pred_train))
                self.AUC_val.append(roc_auc_score(self.val_y, y_pred_val))
                self.train_loss.append(train_loss)
                self.val_loss.append(val_loss)
                self.f1_train.append(
                    f1_score(self.train_y[:, 1], self.make_prediction(y_pred_train[:, 1])))
                self.f1_val.append(
                    f1_score(self.val_y[:, 1], self.make_prediction(y_pred_val[:, 1])))

                if val_loss < self.min_val_loss:
                    self.min_val_loss = val_loss
                    self.best_metrics['epoch'] = epoch
                    self.best_metrics['train'] = get_metrics(
                        y_pred_train, self.train_true_label)
                    self.best_metrics['train']['loss'] = train_loss

                    self.best_metrics['test'] = get_metrics(
                        y_pred_val, self.test_true_label)
                    self.best_metrics['test']['loss'] = val_loss

        self.history = LossHistory(X_train, y_train, X_test, y_test)

    def get_flops(self):

        run_meta_data = tf.RunMetadata()
        flop_opts = tf.profiler.ProfileOptionBuilder.float_operation()

        conv_flops = tf.profiler.profile(graph=K.get_session(
        ).graph, run_meta=run_meta_data, cmd='op', options=flop_opts)
        self.flops = conv_flops.total_float_ops
        print(self.flops)

    # TODO: convert to pytorch training
    def train_model(self, X_train, y_train, X_test, y_test, print_option=0, verbose=2):

        # self.build_loss_history(X_train, y_train, X_test, y_test)
        # self.model.fit(X_train, y_train,
        #                batch_size=self.batch_size,
        #                nb_epoch=self.nb_epoch,
        #                validation_data=(X_test, y_test),
        #                class_weight=self.class_weights, verbose=verbose
        #                , callbacks=[self.history]
        #                )

        # self.get_lastMpercent_loss()

        # if print_option == 1:
        #     print(self.last_Mpercent_epoch_val_loss)

        # add callback features later
        train_loader = X_train
        num_epochs = self.nb_epoch
        device = self.device
        self.model = self.model.to(device)

        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % 2000 == 0:
                    print(
                        f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        print('Finished Training')

        # TODO: convert to pytorch cross validation
    def train_n_fold_cross_val(self, Data, label, training_to_all_data_ratio=0.9, n=10, print_option=0, plot_option=0,
                               save_option=0, save_path='results/test1.png', epoch_resolution=100, verbose=2, seed=0, downsample=False):

        nb_samples = Data.shape[0]
        # get the initial random model weights
        w_save = self.model.get_weights()

        class_0 = []
        class_1 = []

        for idx, l in enumerate(label):
            # [0,1] = risky, [1,0] = safe
            if (l == [0., 1.]).all():
                class_0.append(Data[idx])
            elif (l == [1., 0.]).all():
                class_1.append(Data[idx])

        class_0 = np.array(class_0)
        class_1 = np.array(class_1)

        y_0 = [[0., 1.]] * len(class_0)
        y_1 = [[1., 0.]] * len(class_1)

        y_0 = np.array(y_0, dtype=np.float64)
        y_1 = np.array(y_1, dtype=np.float64)

        min_number = min(len(class_0), len(class_1))
        if downsample:
            class_1, y_1 = resample(class_1, y_1, n_samples=min_number)

        for i in tqdm(range(n)):

            X_train, X_test, y_train, y_test = train_test_split(np.concatenate([class_0, class_1], axis=0), np.concatenate(
                [y_0, y_1], axis=0), test_size=1-training_to_all_data_ratio, shuffle=True, stratify=np.concatenate([y_0, y_1], axis=0), random_state=seed)

            # Model weights from the previous training session must be resetted to the initial random values
            self.model.set_weights(w_save)
            self.history = []
            # class weights must be adjusted for the loss function. In case of weird class distr.
            c1 = round(y_train[y_train[:, 1] == 0,
                               :].shape[0] / y_train.shape[0], 2)
            c2 = 1 - c1
            self.class_weights = {0: c2, 1: c1}
            self.train_model(X_train, y_train, X_test, y_test,
                             print_option=print_option, verbose=verbose)

            if plot_option == 1:
                if i == 0:
                    #plt.plot(self.history.AUC_train[0::epoch_resolution], 'r--')
                    plt.plot(self.history.f1_train[0::epoch_resolution], 'r--')
                #plt.plot(self.history.AUC_val[0::epoch_resolution], 'g')
                plt.plot(self.history.f1_val[0::epoch_resolution], 'g')

            self.m_fold_cross_val_results.append(
                self.last_Mpercent_epoch_val_loss)
        if plot_option == 1:
            if save_option == 1:
                plt.savefig(save_path)
            else:
                plt.show()
        plt.close()
        return self.history.best_metrics

    def get_lastMpercent_loss(self, m=0.1):

        index = int(self.nb_epoch*m)
        self.last_Mpercent_epoch_val_loss = sum(
            self.history.AUC_val[index:])/len(self.history.AUC_val[index:])

    def plot_auc(self, epoch_resolution=1, option='AUC_v_epoch'):

        if option == 'AUC_v_epoch':
            ep = np.arange(0, self.nb_epoch + 1, epoch_resolution)
            plt.plot(ep, self.history.AUC_train[0::epoch_resolution],
                     'r--', ep, self.history.AUC_val[0::epoch_resolution], 'g')
            plt.show()
        elif option == 'loss_v_epoch':
            plt.plot(self.history.train_loss)
            plt.plot(self.history.val_loss)
            plt.show()
        else:
            ep = np.arange(0, self.nb_epoch + 1, epoch_resolution)
            plt.plot(ep, self.history.AUC_train[0::epoch_resolution],
                     'r--', ep, self.history.AUC_val[0::epoch_resolution], 'g')
            plt.show()

            plt.plot(self.history.train_loss)
            plt.plot(self.history.val_loss)
            plt.show()

    @staticmethod
    def split_training_data(input_data, label, training_to_all_data_ratio=0.9):
        # todo
        nb_samples = input_data.shape[0]
        rand_indexes = list(range(0, nb_samples))
        random.shuffle(rand_indexes)

        X_train = input_data[rand_indexes[0:int(
            nb_samples * training_to_all_data_ratio)], :]
        y_train = label[rand_indexes[0:int(
            nb_samples * training_to_all_data_ratio)], :]
        X_test = input_data[rand_indexes[int(
            nb_samples * training_to_all_data_ratio):], :]
        y_test = label[rand_indexes[int(
            nb_samples * training_to_all_data_ratio):], :]

        return X_train, y_train, X_test, y_test

    def set_class_weights(self, y_train):
        c1 = round(y_train[y_train[:, 1] == 0,
                           :].shape[0] / y_train.shape[0], 2)
        c2 = 1 - c1
        self.class_weights = {0: c2, 1: c1}

    def build_transfer_LSTM_model(self, input_shape, lr=1e-6, decay=1e-5):

        model = nn.Sequential(
            nn.GRU(input_size=input_shape[-1], hidden_size=100),
            nn.Linear(in_features=100, out_features=2),
            nn.Softmax(),
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, lr_decay=decay)

        self.loss_func = loss_fn
        self.optimizer = optimizer
        self.model = model

    # TODO: convert to pytorch
    def build_transfer_LSTM_model2(self, input_shape, lr=1e-6, decay=1e-5):

        model = nn.Sequential(
            nn.LSTM(input_size=input_shape[-1], hidden_size=512, num_layers=2),
            nn.Dropout(0.8),
            nn.Linear(in_features=512, out_features=1000),
            nn.Dropout(0.8),
            nn.Linear(in_features=1000, out_features=200),
            nn.Linear(in_features=200, out_features=2),
            nn.Softmax(),
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, lr_decay=decay)

        self.loss_func = loss_fn
        self.optimizer = optimizer
        self.model = model

    # TODO: find out if LSTM can be used in sequential model
    def build_transfer_LSTM_model3(self, input_shape, lr=1e-6, decay=1e-5):

        model = nn.Sequential(
            # TODO: add recurrent dropout
            nn.LSTM(input_size=input_shape[-1], hidden_size=512, dropout=0.5),
            nn.LSTM(input_size=512, hidden_size=512),
            nn.Linear(in_features=512, out_features=1000),
            nn.Linear(in_features=1000, out_features=200),
            nn.Linear(in_features=200, out_features=2),
            nn.Softmax(),
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, lr_decay=decay)

        self.loss_func = loss_fn
        self.optimizer = optimizer
        self.model = model

    def build_cnn_to_lstm_model(self, input_shape, lr=1e-6, decay=1e-5):

        # CNN layers
        layer_1 = TimeDistributed(nn.Conv2d(
            in_channels=input_shape[-1], out_channels=16, kernel_size=3, stride=3))
        activ_1 = TimeDistributed(nn.ReLU())(layer_1)
        output_size = feature_map_size(
            input_size=input_shape[0], kernel_size=3, stride=3)

        layer_2 = TimeDistributed(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=3))(activ_1)
        activ_2 = TimeDistributed(nn.ReLU())(layer_2)
        output_size = feature_map_size(
            input_size=output_size, kernel_size=3, stride=3)

        layer_3 = TimeDistributed(nn.MaxPool2d(kernel_size=(2, 2)))(activ_2)
        activ_3 = TimeDistributed(nn.Dropout(0.2))(layer_3)
        output_size = feature_map_size(input_size=output_size, kernel_size=2)

        layer_4 = TimeDistributed(nn.Flatten())(activ_3)
        output_size = num_flat_features(output_size)

        # FC layers
        layer_5 = TimeDistributed(
            nn.Linear(in_features=output_size, out_features=200))(layer_4)

        layer_6 = TimeDistributed(
            nn.Linear(in_features=200, out_features=50))(layer_5)

        # LSTM layer TODO: verify it is correct
        layer_7, (hn, cn) = nn.LSTM(input_size=50, hidden_size=20)(layer_6)

        # Classification layer
        layer_8 = nn.Linear(in_features=20, out_features=2)(
            layer_7)  # confirm input features
        activ_8 = nn.Softmax()(layer_8)

        model = activ_8
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, lr_decay=decay)

        self.loss_func = loss_fn
        self.optimizer = optimizer
        self.model = model

    def build_cnn_model(self, input_shape, lr=1e-6, decay=1e-5):

        # define model will refactor once tested
        layer = nn.Conv2d(
            in_channels=input_shape[-1], out_channels=32, kernel_size=(5, 5), stride=(1, 1))
        layer = nn.ReLU()(layer)
        output_size = feature_map_size(
            input_size=input_shape[0], kernel_size=5)

        layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(layer)
        output_size = feature_map_size(
            input_size=output_size, kernel_size=2, stride=2)

        layer = nn.Conv2d(in_channels=output_size,
                          out_channels=64, kernel_size=(5, 5))(layer)
        layer = nn.ReLU()(layer)
        output_size = feature_map_size(
            input_size=output_size, kernel_size=5, stride=2)

        layer = nn.MaxPool2d(kernel_size=(2, 2))(layer)
        output_size = feature_map_size(input_size=output_size, kernel_size=2)

        layer = nn.Flatten()(layer)
        output_size = num_flat_features(output_size)

        layer = nn.Linear(in_features=output_size, out_features=1000)(layer)
        layer = nn.ReLU()(layer)
        layer = nn.Linear(in_features=1000, out_features=2)(layer)
        layer = nn.Softmax()(layer)

        model = layer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, lr_decay=decay)

        self.loss_func = loss_fn
        self.optimizer = optimizer
        self.model = model

    # TODO: double check
    def build_transfer_ResNet_to_LSTM(self, input_shape, lr=1e-6, decay=1e-5):

        # TODO: feature sequence missing time distributed & double check dimensions
        backbone_model = models.resnet50(pretrained=True)
        backbone_model.fc = nn.Linear(num_ftrs, input_shape[-1])

        # debug
        feature_sequences = TimeDistributed(backbone_model)(input_sequences)

        model = nn.Sequential(
            feature_sequences,
            nn.LSTM(input_size=input_shape[-1], hidden_size=20),
            nn.Linear(in_features=20, out_features=2),
            nn.Softmax(),
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, lr_decay=decay)

        self.loss_func = loss_fn
        self.optimizer = optimizer
        self.model = model

    # determine output feature map size for convolution layers
    @staticmethod
    def feature_map_size(input_size, kernel_size, padding=0, stride=1):
        output_size = 1 + (input_size - kernel_size + 2*padding)/stride
        return output_size

    # determine output dimension after flatten operation
    @staticmethod
    def num_flat_features(feature_map):
        # all dimensions except the batch dimension
        size = feature_map.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    # https://www.jianshu.com/p/223e13ce35a2
    class TimeDistributed(nn.Module):
        def __init__(self, module, batch_first=False):
            super(TimeDistributed, self).__init__()
            self.module = module
            self.batch_first = batch_first

        def forward(self, x):
            if len(x.size()) <= 2:
                return self.module(x)

            # reshape input data --> (samples * timesteps, input_size)
            # squash timesteps
            x_reshaped = x.contiguous().view(-1, x.size(-1))
            y = self.module(x_reshaped)

            # We have to reshape Y
            if self.batch_first:
                # (samples, timesteps, output_size)
                y = y.contiguous().view(x.size(0), -1, y.size(-1))
            else:
                # (timesteps, samples, output_size)
                y = y.contiguous().view(-1, x.size(1), y.size(-1))
            return y
