import random, cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from metrics import *


class Trainer:
    def __init__(self, config):
        self.config = config
        self.toGPU = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.config.device)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.best_val_loss = 99999
        self.best_epoch = 0
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.best_val_confusion = []
        self.best_val_f1 = 0
        self.best_val_mcc = -1.0
        self.best_val_acc_balanced = 0
        self.unique_clips = {}
        self.log = False # for logging to wandb

    def build_model(self, model):
        self.model = model.to(self.config.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        if self.class_weights.shape[0] < 2:
            self.loss_func = torch.nn.CrossEntropyLoss()
        else:    
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.device))
    
    def reset_weights(self, model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def init_dataset(self, data, label, clip_name):
        nb_samples = data.shape[0] # 574 samples
        class_0 = []
        class_1 = []
        class_0_clip_name = []
        class_1_clip_name = []

        # Store driving categories and their frequencies
        for clip in clip_name:
            category = clip.split('_')[-1]
            if category in self.unique_clips:
                self.unique_clips[category] += 1
            else:
                self.unique_clips[category] = 1
        print('Total dataset breakdown: {}'.format(self.unique_clips))        

        # Remove default_all category if more than one category
        if len(self.unique_clips.keys()) > 1:
            index = np.argwhere(clip_name == 'default_all')
            clip_name = np.delete(clip_name, index)

        for idx, l in enumerate(label):
            # (data, label) -> (class_0, y_0) non-risky, (class_1, y_1) risky
            if (l == [1.0, 0.0]).all(): 
                class_0.append(data[idx])
                class_0_clip_name.append(clip_name[idx])
            elif (l == [0.0, 1.0]).all(): 
                class_1.append(data[idx])
                class_1_clip_name.append(clip_name[idx])
            
        y_0 = [0] * len(class_0)
        y_1 = [1] * len(class_1)
        
        self.class_0 = np.array(class_0)
        self.class_1 = np.array(class_1)
        self.class_0_clip_name = np.array(class_0_clip_name)
        self.class_1_clip_name = np.array(class_1_clip_name)
        self.y_0 = np.array(y_0, dtype=np.float64)
        self.y_1 = np.array(y_1, dtype=np.float64)

        # balance the dataset
        min_number = min(len(class_0), len(class_1))
        if self.config.downsample: 
            if len(class_0) > len(class_1):
                self.class_0, self.class_0_clip_name, self.y_0 = resample(class_0, class_0_clip_name, y_0, n_samples=min_number, random_state=self.config.seed);
            else:
                self.class_1, self.class_1_clip_name, self.y_1 = resample(class_1, class_1_clip_name, y_1, n_samples=min_number, random_state=self.config.seed);
        self.split_dataset()

    def split_dataset(self):
        self.training_x, self.testing_x, self.training_clip_name, self.testing_clip_name, self.training_y, self.testing_y  = train_test_split(
            np.concatenate([self.class_0, self.class_1], axis=0),
            np.concatenate([self.class_0_clip_name, self.class_1_clip_name], axis=0),
            np.concatenate([self.y_0, self.y_1], axis=0),
            test_size=1-self.config.train_ratio,
            shuffle=True,
            stratify=np.concatenate([self.y_0, self.y_1], axis=0),
            random_state=self.config.seed,
        )
        self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.training_y), self.training_y))
        if self.config.n_folds <= 1:
            print("Number of Training Sequences Included: ", len(self.training_x))
            print("Number of Testing Sequences Included: ", len(self.testing_x))
            print("Num of Training Labels in Each Class: " + str(np.unique(self.training_y, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            print("Num of Testing Labels in Each Class: " + str(np.unique(self.testing_y, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))

    def train_n_fold_cross_val(self):
        # KFold cross validation with similar class distribution in each fold
        skf = StratifiedKFold(n_splits=self.config.n_folds)
        X = np.append(self.training_x, self.testing_x, axis=0)
        clip_name = np.append(self.training_clip_name, self.testing_clip_name, axis=0)
        y = np.append(self.training_y, self.testing_y, axis=0)

        # self.results stores average metrics for the the n_folds
        self.results = {}
        self.fold = 1

        # Split training and testing data based on n_splits (Folds)
        for train_index, test_index in skf.split(X, y):
            self.training_x, self.testing_x, self.training_clip_name, self.testing_clip_name, self.training_y, self.testing_y = None, None, None, None, None, None #clear vars to save memory
            X_train, X_test = X[train_index], X[test_index]
            clip_train, clip_test = clip_name[train_index], clip_name[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(y_train), y_train))

            # Update dataset
            self.training_x = X_train
            self.testing_x  = X_test
            self.training_clip_name = clip_train
            self.testing_clip_name = clip_test
            self.training_y = y_train
            self.testing_y  = y_test

            print('\nFold {}'.format(self.fold))
            print("Number of Training Sequences Included: ", len(X_train))
            print("Number of Testing Sequences Included: ",  len(X_test))
            print("Num of Training Labels in Each Class: " + str(np.unique(self.training_y, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            print("Num of Testing Labels in Each Class: "  + str(np.unique(self.testing_y, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
           
            self.best_val_loss = 99999
            self.train_model()
            self.log = True
            categories_train, categories_test, metrics = self.eval_model(self.fold)
            self.update_cross_valid_metrics(categories_train, categories_test, metrics)
            self.log = False

            if self.fold != self.config.n_folds:
                self.reset_weights(self.model)
                del self.optimizer
                self.build_model(self.model)
                
            self.fold += 1            
        del self.results

    def train_model(self):
        tqdm_bar = tqdm(range(self.config.epochs))
        for epoch_idx in tqdm_bar: # iterate through epoch   
            acc_loss_train = 0
            permutation = np.random.permutation(len(self.training_x)) # shuffle dataset before each epoch
            self.model.train()

            for i in range(0, len(self.training_x), self.config.batch_size): # iterate through batches of the dataset
                batch_index = i + self.config.batch_size if i + self.config.batch_size <= len(self.training_x) else len(self.training_x)
                indices = permutation[i:batch_index]
                batch_x, batch_y = self.training_x[indices], self.training_y[indices]
                batch_x, batch_y = self.toGPU(batch_x, torch.float32), self.toGPU(batch_y, torch.long)
                output = self.model.forward(batch_x).view(-1, 2)
                loss_train = self.loss_func(output, batch_y)
                loss_train.backward()
                acc_loss_train += loss_train.detach().cpu().item() * len(indices)
                self.optimizer.step()
                del loss_train

            acc_loss_train /= len(self.training_x)
            tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
            
            # no cross validation 
            if epoch_idx % self.config.test_step == 0:
                self.eval_model(epoch_idx)

    def model_inference(self, X, y, clip_name):
        labels = torch.LongTensor().to(self.config.device)
        outputs = torch.FloatTensor().to(self.config.device)
        # Dictionary storing (output, label) pair for all driving categories
        categories = dict.fromkeys(self.unique_clips)
        for key, val in categories.items():
            categories[key] = {'outputs': outputs, 'labels': labels}
        batch_size = self.config.batch_size # NOTE: set to 1 when profiling or calculating inference time.
        acc_loss = 0
        inference_time = 0
        prof_result = ""

        with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
            with torch.no_grad():
                self.model.eval()

                for i in range(0, len(X), batch_size): # iterate through subsequences
                    batch_index = i + batch_size if i + batch_size <= len(X) else len(X)
                    batch_x, batch_y, batch_clip_name = X[i:batch_index], y[i:batch_index], clip_name[i:batch_index]
                    batch_x, batch_y = self.toGPU(batch_x, torch.float32), self.toGPU(batch_y, torch.long)
                    #start = torch.cuda.Event(enable_timing=True)
                    #end =  torch.cuda.Event(enable_timing=True)
                    #start.record()
                    output = self.model.forward(batch_x).view(-1, 2)
                    #end.record()
                    #torch.cuda.synchronize()
                    inference_time += 0#start.elapsed_time(end)
                    loss_test = self.loss_func(output, batch_y)
                    acc_loss += loss_test.detach().cpu().item() * len(batch_y)
                    # store output, label statistics
                    self.update_categorical_outputs(categories, output, batch_y, batch_clip_name)

        # calculate one risk score per sequence (this is not implemented for each category)
        sum_seq_len = 0
        num_risky_sequences = 0
        num_safe_sequences = 0
        correct_risky_seq = 0
        correct_safe_seq = 0
        incorrect_risky_seq = 0
        incorrect_safe_seq = 0
        sequences = len(categories['all']['labels'])
        for indices in range(sequences):
            seq_output = categories['all']['outputs'][indices]
            label = categories['all']['labels'][indices]
            pred = torch.argmax(seq_output)
            
            # risky clip
            if label == 1:
                num_risky_sequences += 1
                sum_seq_len += seq_output.shape[0]
                correct_risky_seq += self.correctness(label, pred)
                incorrect_risky_seq += self.correctness(label, pred)
            # non-risky clip
            elif label == 0:
                num_safe_sequences += 1
                incorrect_safe_seq += self.correctness(label, pred)
                correct_safe_seq += self.correctness(label, pred)
        
        avg_risky_seq_len = sum_seq_len / num_risky_sequences # sequence length for comparison with the prediction frame metric. 
        seq_tpr = correct_risky_seq / num_risky_sequences
        seq_fpr = incorrect_safe_seq / num_safe_sequences
        seq_tnr = correct_safe_seq / num_safe_sequences
        seq_fnr = incorrect_risky_seq / num_risky_sequences
        if prof != None:
            prof_result = prof.key_averages().table(sort_by="cuda_time_total")

        return  categories, \
                acc_loss/len(X), \
                avg_risky_seq_len, \
                inference_time, \
                prof_result, \
                seq_tpr, \
                seq_fpr, \
                seq_tnr, \
                seq_fnr

    def eval_model(self, current_epoch=None):
        metrics = {}
        categories_train, \
        acc_loss_train, \
        train_avg_seq_len, \
        train_inference_time, \
        train_profiler_result, \
        seq_tpr, seq_fpr, seq_tnr, seq_fnr = self.model_inference(self.training_x, self.training_y, self.training_clip_name) 

        # Collect metrics from all driving categories
        for category in self.unique_clips.keys():
            if category == 'all':
                metrics['train'] = get_metrics(categories_train['all']['outputs'], categories_train['all']['labels'])
                metrics['train']['loss'] = acc_loss_train
                metrics['train']['avg_seq_len'] = train_avg_seq_len
                metrics['train']['seq_tpr'] = seq_tpr
                metrics['train']['seq_tnr'] = seq_tnr
                metrics['train']['seq_fpr'] = seq_fpr
                metrics['train']['seq_fnr'] = seq_fnr
            else:
                metrics['train'][category] = get_metrics(categories_train[category]['outputs'], categories_train[category]['labels'])

        categories_test, \
        acc_loss_test, \
        val_avg_seq_len, \
        test_inference_time, \
        test_profiler_result, \
        seq_tpr, seq_fpr, seq_tnr, seq_fnr = self.model_inference(self.testing_x, self.testing_y, self.testing_clip_name) 

        # Collect metrics from all driving categories
        for category in self.unique_clips.keys():
            if category == 'all':
                metrics['test'] = get_metrics(categories_test['all']['outputs'], categories_test['all']['labels'])
                metrics['test']['loss'] = acc_loss_test
                metrics['test']['avg_seq_len'] = val_avg_seq_len
                metrics['test']['seq_tpr'] = seq_tpr
                metrics['test']['seq_tnr'] = seq_tnr
                metrics['test']['seq_fpr'] = seq_fpr
                metrics['test']['seq_fnr'] = seq_fnr
                metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / ((len(self.training_y) + len(self.testing_y))*5)
            else:
                metrics['test'][category] = get_metrics(categories_test[category]['outputs'], categories_test[category]['labels'])

        
        print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], "mcc:", metrics['train']['mcc'], \
              "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], "mcc:", metrics['test']['mcc'])

        self.update_best_metrics(metrics, current_epoch)
        metrics['best_epoch'] = self.best_epoch
        metrics['best_val_loss'] = self.best_val_loss
        metrics['best_val_acc'] = self.best_val_acc
        metrics['best_val_auc'] = self.best_val_auc
        metrics['best_val_conf'] = self.best_val_confusion
        metrics['best_val_f1'] = self.best_val_f1
        metrics['best_val_mcc'] = self.best_val_mcc
        metrics['best_val_acc_balanced'] = self.best_val_acc_balanced
        
        if self.config.n_folds <= 1 or self.log:  
            self.log2wandb(metrics)

        return categories_train, categories_test, metrics


    #automatically save the model and metrics with the lowest validation loss
    def update_best_metrics(self, metrics, current_epoch):
        if metrics['test']['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['test']['loss']
            self.best_epoch = current_epoch if current_epoch != None else self.config.epochs
            self.best_val_acc = metrics['test']['acc']
            self.best_val_auc = metrics['test']['auc']
            self.best_val_confusion = metrics['test']['confusion']
            self.best_val_f1 = metrics['test']['f1']
            self.best_val_mcc = metrics['test']['mcc']
            self.best_val_acc_balanced = metrics['test']['balanced_acc']
            #self.save_model()
    
    def update_cross_valid_metrics(self, categories_train, categories_test, metrics):
        '''
            Stores cross-validation metrics for all driving categories
        '''
        datasets = ['train', 'test']
        if self.fold == 1:
            for dataset in datasets:
                categories = categories_train if dataset == 'train' else categories_test
                for category in self.unique_clips.keys():
                    if category == 'all':
                        self.results['outputs'+'_'+dataset] = categories['all']['outputs']
                        self.results['labels'+'_'+dataset] = categories['all']['labels']
                        self.results[dataset] = metrics[dataset]
                        self.results[dataset]['loss'] = metrics[dataset]['loss']
                        self.results[dataset]['avg_seq_len']  = metrics[dataset]['avg_seq_len'] 
                        
                        # Best results
                        self.results['avg_inf_time']  = metrics['avg_inf_time']
                        self.results['best_epoch']    = metrics['best_epoch']
                        self.results['best_val_loss'] = metrics['best_val_loss']
                        self.results['best_val_acc']  = metrics['best_val_acc']
                        self.results['best_val_auc']  = metrics['best_val_auc']
                        self.results['best_val_conf'] = metrics['best_val_conf']
                        self.results['best_val_f1']   = metrics['best_val_f1']
                        self.results['best_val_mcc']  = metrics['best_val_mcc']
                        self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']
                    else:
                        self.results[dataset][category]['outputs'] = categories[category]['outputs']
                        self.results[dataset][category]['labels'] = categories[category]['labels']

        else:
            for dataset in datasets:
                categories = categories_train if dataset == 'train' else categories_test
                for category in self.unique_clips.keys():
                    if category == 'all':
                        self.results['outputs'+'_'+dataset] = torch.cat((self.results['outputs'+'_'+dataset], categories['all']['outputs']), dim=0)
                        self.results['labels'+'_'+dataset]  = torch.cat((self.results['labels'+'_'+dataset], categories['all']['labels']), dim=0)
                        self.results[dataset]['loss'] = np.append(self.results[dataset]['loss'], metrics[dataset]['loss'])
                        self.results[dataset]['avg_seq_len']  = np.append(self.results[dataset]['avg_seq_len'], metrics[dataset]['avg_seq_len'])
                        
                        # Best results
                        self.results['avg_inf_time']  = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])
                        self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
                        self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
                        self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
                        self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
                        self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
                        self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
                        self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
                        self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
                    else:
                        self.results[dataset][category]['outputs'] = torch.cat((self.results[dataset][category]['outputs'], categories[category]['outputs']), dim=0)
                        self.results[dataset][category]['labels']  = torch.cat((self.results[dataset][category]['labels'], categories[category]['labels']), dim=0)
            
        # Log final averaged results
        if self.fold == self.config.n_folds:
            final_metrics = {}
            for dataset in datasets:
                for category in self.unique_clips.keys():
                    if category == 'all':
                        final_metrics[dataset] = get_metrics(self.results['outputs'+'_'+dataset], self.results['labels'+'_'+dataset])
                        final_metrics[dataset]['loss'] = np.average(self.results[dataset]['loss'])
                        final_metrics[dataset]['avg_seq_len'] = np.average(self.results[dataset]['avg_seq_len'])

                        # Best results
                        final_metrics['avg_inf_time']  = np.average(self.results['avg_inf_time'])
                        final_metrics['best_epoch']    = np.average(self.results['best_epoch'])
                        final_metrics['best_val_loss'] = np.average(self.results['best_val_loss'])
                        final_metrics['best_val_acc']  = np.average(self.results['best_val_acc'])
                        final_metrics['best_val_auc']  = np.average(self.results['best_val_auc'])
                        final_metrics['best_val_conf'] = self.results['best_val_conf']
                        final_metrics['best_val_f1']   = np.average(self.results['best_val_f1'])
                        final_metrics['best_val_mcc']  = np.average(self.results['best_val_mcc'])
                        final_metrics['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
                    else: 
                        final_metrics[dataset][category] = get_metrics(self.results[dataset][category]['outputs'], self.results[dataset][category]['labels'])

            print('\nFinal Averaged Results')
            print("\naverage train loss: " + str(final_metrics['train']['loss']) + ", average acc:", final_metrics['train']['acc'], final_metrics['train']['confusion'], final_metrics['train']['auc'], \
                "\naverage test loss: " +  str(final_metrics['test']['loss']) + ", average acc:", final_metrics['test']['acc'],  final_metrics['test']['confusion'], final_metrics['test']['auc'])

            self.log2wandb(final_metrics)
            
            # final combined results and metrics
            return self.results['outputs_train'], self.results['labels_train'], self.results['outputs_test'], self.results['labels_test'], final_metrics

    # Utilities
    def update_categorical_outputs(self, categories, outputs, labels, clip_name):
        '''
            Aggregates output, label pairs for every driving category
        '''
        n = len(clip_name)
        for i in range(n):
            category = clip_name[i].split('_')[-1]
            # FIXME: probably better way to do this
            if category in categories:
                categories[category]['outputs'] = torch.cat([categories[category]['outputs'], torch.unsqueeze(outputs[i], dim=0)], dim=0)
                categories[category]['labels'] = torch.cat([categories[category]['labels'], torch.unsqueeze(labels[i], dim=0)], dim=0)
            # multi category
            if category != 'all': 
                category = 'all'
                categories[category]['outputs'] = torch.cat([categories[category]['outputs'], torch.unsqueeze(outputs[i], dim=0)], dim=0)
                categories[category]['labels'] = torch.cat([categories[category]['labels'], torch.unsqueeze(labels[i], dim=0)], dim=0)

        # reshape outputs
        for k, v in categories.items():
            categories[k]['outputs'] = categories[k]['outputs'].reshape(-1, 2)
   
    def preprocess_batch(self, x):
        '''
            Apply normalization preprocess to all data
        '''
        b = []
        for batch in x:
            d = []
            for data in batch:
                data = np.moveaxis(data, 0, -1) # move channels to last_dim
                d.append(preprocess_image(data))
            b.append(torch.cat(d, axis=0))
        return torch.stack(b, dim=0).type(torch.float32)

    def correctness(self, output, pred):
        return 1 if output == pred else 0

    def log2wandb(self, metrics):
        '''
            Log metrics from all driving categories
        '''
        for category in self.unique_clips.keys():
            if category == 'all':
                log_wandb(self.config.wandb, metrics)
            else:
                log_wandb_categories(self.config.wandb, metrics, id=category)