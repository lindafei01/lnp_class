import numpy as np
import pandas as pd
import sys
import pickle as pkl
import random
import os
import pdb
from tqdm import tqdm
from threading import Thread, Lock
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import argparse

from unicore.modules import init_bert_params
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord,
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from MolCLR_dataset import MolTestDatasetWrapper, MolTestDatasetWrapper_smiles
from gcn import GCN
from ginet import GINet
from sklearn.utils import resample
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.pipeline import Pipeline
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import itertools
import torch.optim.lr_scheduler as lr_scheduler
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import auc

def true_positive(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum()

def false_positive(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 1)).sum()

def true_negative(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 0)).sum()

def false_negative(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 0)).sum()

class MLPClassifier(nn.Module):
    def __init__(self, hidden_layer_sizes=(100,), activation=nn.ReLU, solver='adam', early_stopping=True,
                 patience=30, warmup_epochs=30, lr=0.001, num_epochs=100000, batch_size=32, class_num=1, device=None, args=None):

        super(MLPClassifier, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.early_stopping = early_stopping
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.class_num = class_num
        self.best_model = None
        self.last_model = None
        self.device = device
        
        self.run = args.run

    def fit(self, X_train, y_train):
        # scaler = nn.BatchNorm1d(X_train.shape[1]).to(self.device)
        # scaler.eval()  
        # with torch.no_grad():
        #     X_train = scaler(X_train)
        #     X_val = scaler(X_val)

        self.hidden_layers = []
        input_dim = X_train.shape[1]
        for layer_size in self.hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(input_dim, layer_size))
            self.hidden_layers.append(nn.ReLU())
            input_dim = layer_size
        
        self.hidden_layers = nn.Sequential(*self.hidden_layers).to(self.device)
        self.output_layer = nn.Linear(input_dim, self.class_num).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)
        # self.model = nn.Sequential(*layers).to(self.device)

        if self.solver == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.solver == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError("unknown solver")
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1, (epoch + 1) / self.warmup_epochs))
        
        criterion = nn.BCELoss()

        # train_dataset = TensorDataset(X_train, y_train)
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train(True)

        import math
        self.number_batches = math.ceil(len(X_train) / self.batch_size)
        
        for epoch in range(self.num_epochs):
            scheduler.step()
            running_loss = 0.0
            for batch_idx in range(self.number_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(X_train))
                
                inputs = X_train[start_idx:end_idx]
                labels = y_train[start_idx:end_idx]
                outputs = self(inputs)
                
                loss = criterion(outputs.squeeze(), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / self.number_batches
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")
            if self.run is not None:
                self.run["train/loss"].append(epoch_loss)
                self.run["train/lr"].append(optimizer.param_groups[0]["lr"])
            
    def forward(self, X):
        for hidden_layer in self.hidden_layers:
            X = hidden_layer(X)
        X = self.output_layer(X)
        output = self.sigmoid(X)
        return output
        
    def predict(self, X, version):
        assert version in ["best", "last"]
        if version == "last":
            self.model.load_state_dict(self.last_model)
        if version == "best" and self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            outputs = self.model(X)
            _, predictions = torch.max(outputs, dim=1)

        return predictions.cpu()

    def evaluate(self, X, label, k):
        with torch.no_grad():
            self.train(False)
            pre = self(X)
            
        prec, rec, thr = precision_recall_curve(label.cpu().numpy(), pre.cpu().numpy())
        fpr, tpr, thr = roc_curve(label.cpu().numpy(), pre.cpu().numpy())
        tn, fp, fn, tp = confusion_matrix(y_pred=pre.round().cpu().numpy(), y_true=label.cpu().numpy()).ravel()
        
        if self.run is not None:
            metric_auc = auc(fpr, tpr)
            metric_aupr = auc(rec, prec)
            metric_acc = accuracy_score(y_pred=pre.round().cpu().numpy(), y_true=label.cpu().numpy())
            metric_f1 = f1_score(y_pred=pre.round().cpu().numpy(), y_true=label.cpu().numpy())
            
            # 计算Top Recall, 我还没有完全s想好
            # 假设您想计算前k个样本的Recall
            top_k_indices = np.argsort(pre.cpu().numpy(), axis=0)[-k:].ravel()
            top_k_label = label[top_k_indices]
            top_k_recall = np.sum(top_k_label.cpu().numpy()) / np.sum(label.cpu().numpy())
            # self.run["eval/top_recall@{}".format(k)] = top_k_recall
        
        return metric_auc, metric_aupr, metric_acc, metric_f1, top_k_recall, tn, fn, tp, fp
        

def train_mlp_from_mixed_dataset_cv(X_primary, y_primary, X_train_related,
                                     y_train_related, X_test_related,
                                     y_test_related, k_folds, args):
    
    param_grid = {
        'hidden_layer_sizes': [(128, 32)],
        'activation': [nn.ReLU],
        'solver': ['adam']
    }

    best_val_recall = -np.inf
    best_val_metric = {}
    best_train_fitting_metric = {}
    train_fitting_metric_last = {}
    best_model = None


    kf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
    

    for params in itertools.product(*param_grid.values()):
        hidden_layer_sizes, activation, solver = params

        accuracy_folds = []
        precision_folds = []
        recall_folds = []
        f1_folds = []
        auroc_folds = []
        tp_folds = []
        fp_folds = []
        tn_folds = []
        fn_folds = []
        
        accuracy_folds_train_fitting = []
        precision_folds_train_fitting = []
        recall_folds_train_fitting = []
        f1_folds_train_fitting = []
        auroc_folds_train_fitting = []
        tp_folds_train_fitting = []
        fp_folds_train_fitting = []
        tn_folds_train_fitting = []
        fn_folds_train_fitting = []
        
        accuracy_folds_train_fitting_last = []
        precision_folds_train_fitting_last = []
        recall_folds_train_fitting_last = []
        f1_folds_train_fitting_last = []
        auroc_folds_train_fitting_last = []
        tp_folds_train_fitting_last = []
        fp_folds_train_fitting_last = []
        tn_folds_train_fitting_last = []
        fn_folds_train_fitting_last = []


        for train_index, val_index in kf.split(X_primary.cpu().numpy(), y_primary.cpu().numpy()):
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation,
                                solver=solver,
                                device=args.device).to(args.device)

            X_train_fold, X_val_fold = X_primary[train_index], X_primary[val_index]
            y_train_fold, y_val_fold = y_primary[train_index], y_primary[val_index]
            
            X_train_combined = torch.cat((X_train_fold, X_train_related), dim=0)
            y_train_combined = torch.cat((y_train_fold, y_train_related), dim=0)

            
            if args.resample:
                ros = RandomOverSampler()
                X_train_combined, y_train_combined = ros.fit_resample(X_train_combined.cpu().numpy(), y_train_combined.cpu().numpy())
                X_train_combined = torch.from_numpy(X_train_combined).to(args.device)
                y_train_combined = torch.from_numpy(y_train_combined).to(args.device)
                


            rng = np.random.default_rng()
            random_indices = rng.permutation(len(X_train_combined))
            X_train_combined = X_train_combined[random_indices]
            y_train_combined = y_train_combined[random_indices]


            model.fit(X_train_combined, y_train_combined, X_val_fold, y_val_fold)

            accuracy, precision, recall, f1, auroc, tp, fp, tn, fn = model.evaluate(X_val_fold, y_val_fold, version="best")
            accuracy_folds.append(accuracy)
            precision_folds.append(precision)
            recall_folds.append(recall)
            f1_folds.append(f1)
            auroc_folds.append(auroc)
            tp_folds.append(tp)
            fp_folds.append(fp)
            tn_folds.append(tn)
            fn_folds.append(fn)
            
            accuracy_train_fitting, precision_train_fitting, \
                recall_train_fitting, f1_train_fitting, auroc_train_fitting, \
                    tp_train_fitting, fp_train_fitting, \
                        tn_train_fitting, fn_train_fitting = model.evaluate(X_train_combined, y_train_combined, version="best")
                        
            accuracy_folds_train_fitting.append(accuracy_train_fitting)
            precision_folds_train_fitting.append(precision_train_fitting)
            recall_folds_train_fitting.append(recall_train_fitting)
            f1_folds_train_fitting.append(f1_train_fitting)
            auroc_folds_train_fitting.append(auroc_train_fitting)
            tp_folds_train_fitting.append(tp_train_fitting)
            fp_folds_train_fitting.append(fp_train_fitting)
            tn_folds_train_fitting.append(tn_train_fitting)
            fn_folds_train_fitting.append(fn_train_fitting)            
            
            accuracy_train_fitting_last, precision_train_fitting_last, \
                recall_train_fitting_last, f1_train_fitting_last, auroc_train_fitting_last, \
                    tp_train_fitting_last, fp_train_fitting_last, \
                        tn_train_fitting_last, fn_train_fitting_last = model.evaluate(X_train_combined, y_train_combined, version="last")
                        
            accuracy_folds_train_fitting_last.append(accuracy_train_fitting_last)
            precision_folds_train_fitting_last.append(precision_train_fitting_last)
            recall_folds_train_fitting_last.append(recall_train_fitting_last)
            f1_folds_train_fitting_last.append(f1_train_fitting_last)
            auroc_folds_train_fitting_last.append(auroc_train_fitting_last)
            tp_folds_train_fitting_last.append(tp_train_fitting_last)
            fp_folds_train_fitting_last.append(fp_train_fitting_last)
            tn_folds_train_fitting_last.append(tn_train_fitting_last)
            fn_folds_train_fitting_last.append(fn_train_fitting_last)            

        if np.mean(recall_folds) > best_val_recall or best_val_recall == -np.inf:
            best_val_metric = {"accuracy": f"{np.mean(accuracy_folds)}±{np.std(accuracy_folds)}",
                               "precision": f"{np.mean(precision_folds)}±{np.std(precision_folds)}",
                               "recall": f"{np.mean(recall_folds)}±{np.std(recall_folds)}",
                               "f1": f"{np.mean(f1_folds)}±{np.std(f1_folds)}",
                               "auroc": f"{np.mean(auroc_folds)}±{np.std(auroc_folds)}",
                               "tp": f"{np.mean(tp_folds)}±{np.std(tp_folds)}",
                               "fp": f"{np.mean(fp_folds)}±{np.std(fp_folds)}",
                               "tn": f"{np.mean(tn_folds)}±{np.std(tn_folds)}",
                               "fn": f"{np.mean(fn_folds)}±{np.std(fn_folds)}"}
            
            best_train_fitting_metric = {"accuracy": f"{np.mean(accuracy_folds_train_fitting)}±{np.std(accuracy_folds_train_fitting)}",
                               "precision": f"{np.mean(precision_folds_train_fitting)}±{np.std(precision_folds_train_fitting)}",
                               "recall": f"{np.mean(recall_folds_train_fitting)}±{np.std(recall_folds_train_fitting)}",
                               "f1": f"{np.mean(f1_folds_train_fitting)}±{np.std(f1_folds_train_fitting)}",
                               "auroc": f"{np.mean(auroc_folds_train_fitting)}±{np.std(auroc_folds_train_fitting)}",
                               "tp": f"{np.mean(tp_folds_train_fitting)}±{np.std(tp_folds_train_fitting)}",
                               "fp": f"{np.mean(fp_folds_train_fitting)}±{np.std(fp_folds_train_fitting)}",
                               "tn": f"{np.mean(tn_folds_train_fitting)}±{np.std(tn_folds_train_fitting)}",
                               "fn": f"{np.mean(fn_folds_train_fitting)}±{np.std(fn_folds_train_fitting)}"}
            
            train_fitting_metric_last = {"accuracy": f"{np.mean(accuracy_folds_train_fitting_last)}±{np.std(accuracy_folds_train_fitting_last)}",
                               "precision": f"{np.mean(precision_folds_train_fitting_last)}±{np.std(precision_folds_train_fitting_last)}",
                               "recall": f"{np.mean(recall_folds_train_fitting)}±{np.std(recall_folds_train_fitting)}",
                               "f1": f"{np.mean(f1_folds_train_fitting_last)}±{np.std(f1_folds_train_fitting_last)}",
                               "auroc": f"{np.mean(auroc_folds_train_fitting_last)}±{np.std(auroc_folds_train_fitting_last)}",
                               "tp": f"{np.mean(tp_folds_train_fitting_last)}±{np.std(tp_folds_train_fitting_last)}",
                               "fp": f"{np.mean(fp_folds_train_fitting_last)}±{np.std(fp_folds_train_fitting_last)}",
                               "tn": f"{np.mean(tn_folds_train_fitting_last)}±{np.std(tn_folds_train_fitting_last)}",
                               "fn": f"{np.mean(fn_folds_train_fitting_last)}±{np.std(fn_folds_train_fitting_last)}"}
            best_model = model

    accuracy, precision, recall, f1, auroc, tp, fp, tn, fn = best_model.evaluate(X_test_related, y_test_related, version="best")
    test_related_metric = {"accuracy": accuracy, "precision": precision,
                           "recall": recall, "f1": f1, "auroc": auroc,
                           "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    return best_model, best_train_fitting_metric, train_fitting_metric_last, best_val_metric, test_related_metric

def train_mlp_from_mixed_dataset(X_primary, y_primary, X_train_related,
                                     y_train_related, X_test_related,
                                     y_test_related, k_folds, args):
    
    param_grid = {
        'hidden_layer_sizes': [(256, 128, 64, 32, 16, 8, 4)],
        'activation': [nn.ReLU],
        'solver': ['adam']
    }

    for params in itertools.product(*param_grid.values()):
        hidden_layer_sizes, activation, solver = params

        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            solver=solver,
                            device=args.device, 
                            args=args).to(args.device)
        
        X_train_combined = torch.cat((X_primary, X_train_related), dim=0)
        y_train_combined = torch.cat((y_primary, y_train_related), dim=0)

        if args.resample:
            ros = RandomOverSampler()
            X_train_combined, y_train_combined = ros.fit_resample(X_train_combined.cpu().numpy(), y_train_combined.cpu().numpy())
            X_train_combined = torch.from_numpy(X_train_combined).to(args.device)
            y_train_combined = torch.from_numpy(y_train_combined).to(args.device)
            
        # rng = np.random.default_rng()
        # random_indices = rng.permutation(len(X_train_combined))
        # X_train_combined = X_train_combined[random_indices]
        # y_train_combined = y_train_combined[random_indices]

        model.fit(X_train_combined, y_train_combined)

        metric_auc, metric_aupr, metric_acc, metric_f1, top_k_recall, tn, fn, tp, fp = model.evaluate(X_test_related, y_test_related, args.k)
        
        metric_auc_fit, metric_aupr_fit, metric_acc_fit, metric_f1_fit, top_k_recall_fit, tn_fit, fn_fit, tp_fit, fp_fit = model.evaluate(X_train_combined, y_train_combined, args.k)
        
        if args.run is not None:       
            args.run["eval/auc"] = metric_auc
            args.run["eval/aupr"] = metric_aupr
            args.run["eval/acc"] = metric_acc
            args.run["eval/f1"] = metric_f1
            args.run[f"eval/top_{args.k}_recall"] = top_k_recall
            args.run["eval/tp"] = tp
            args.run["eval/fp"] = fp
            args.run["eval/tn"] = tn
            args.run["eval/fn"] = fn

            args.run["eval/auc_fit"] = metric_auc_fit
            args.run["eval/aupr_fit"] = metric_aupr_fit
            args.run["eval/acc_fit"] = metric_acc_fit
            args.run["eval/f1_fit"] = metric_f1_fit
            args.run[f"eval/top_{args.k}_recall_fit"] = top_k_recall_fit
            args.run["eval/tp_fit"] = tp_fit
            args.run["eval/fp_fit"] = fp_fit
            args.run["eval/tn_fit"] = tn_fit
            args.run["eval/fn_fit"] = fn_fit
        
        
        


