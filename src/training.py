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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.decomposition import PCA
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
from utils import get_MolCLR_data_loader, get_UniMol_data_loader
from mixed_training import MLPClassifier
from unimol_encoder import UniMolEncoder
from imblearn.over_sampling import RandomOverSampler



def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[1]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[2]


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[3]


scoring = {'accuracy': 'accuracy', 'precision': 'precision',
           'recall': 'recall', 'f1': 'f1',
           'tp': make_scorer(tp), 'fp': make_scorer(fp),
           'tn': make_scorer(tn), 'fn': make_scorer(fn)}

def convert_smiles_list_to_data_loader(remove_hydrogen, smiles_list: list):
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        smiles_dataset = KeyDataset(data_list, "smiles")
        # label_dataset = KeyDataset(data_list, "label")
        dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
        dataset = AtomTypeDataset(data_list, dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", remove_hydrogen, False)
        dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
        coord_dataset = KeyDataset(dataset, "coordinates")
        src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        return NestedDictionaryDataset({
            "input": {
                "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(), ),
                "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0, ),
                "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0, ),
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0, ),
                "smiles": RawArrayDataset(smiles_dataset),
            },
            # "target": {
            #     "label": RawLabelDataset(label_dataset),
            # }
        })

    def calculate_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)

            molecule = Chem.MolFromSmiles(smiles)
            try:
                molecule = AllChem.AddHs(molecule)
            except:
                print("MolFromSmiles error", smiles)
                mutex.acquire()
                mutex.release()
                continue
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()

            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()

    mutex = Lock()
    global smiles_to_conformation_dict
    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_3D_structure_, args=(smiles_list,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    batch_size = 1

    data_list = []
    for smiles in smiles_list:
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
            }
            data_list.append(data_item)

    data = [data_list[i] for i in range(len(data_list))]

    dataset = convert_data_list_to_dataset_(data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collater)

    return data_loader

def get_UniMol_embedding(data_path, model_choice, smiles_list: list, device):
    if model_choice == "unimol_all_h":
        remove_hydrogen = False
    elif model_choice == "unimol_no_h":
        remove_hydrogen = True
    else:
        raise ValueError("unknown embedding type")

    data_loader = convert_smiles_list_to_data_loader(remove_hydrogen, smiles_list=smiles_list)

    with torch.no_grad():
        model = UniMolEncoder(device=device, remove_hydrogen=remove_hydrogen)  # without shuffle
        model.to(device)

        representations = []
        embedded_smiles = []
        for batch in tqdm(data_loader):
            try:
                batch_smiles, batch_representations = model(batch)
                representations.append(batch_representations)
                embedded_smiles.extend(batch_smiles)
            except:
                continue

    with open(f"data/result/{model_choice}_invalid_smiles.txt", "w") as file:
        for smiles in smiles_list:
            if smiles not in embedded_smiles:
                file.write(smiles + "\n")

    data = pd.read_csv(data_path)
    y = []
    for smiles in embedded_smiles:
        row = data[data['SMILES'] == smiles]
        lnp_class = row.iloc[0]['LNP Class']
        y.append(lnp_class)

    for idx, smiles in enumerate(embedded_smiles):
        descs = torch.tensor(data.loc[data['SMILES'] == smiles, 'desc_ABC':].values.tolist()).to(device)
        representations[idx] = torch.cat((representations[idx], descs), dim=1)

    return torch.cat(representations, dim=0), torch.Tensor(y).to(device)


def get_MolCLR_embedding(data_path, smiles_list: list, model_choice, device):
    dataset = MolTestDatasetWrapper_smiles(smiles_list=smiles_list)
    dataloader, embeddable = dataset.get_fulldata_loader()
    if not all(not value for value in embeddable.values()):
        assert model_choice in ["gin", "gcn", "gin_dc", "gcn_dc", "gin_agile", "gin_agile_dc"]
        
        with torch.no_grad():
            if model_choice in ["gin", "gin_dc", "gin_agile", "gin_agile_dc"]:
                model = GINet()
                try:
                    state_dict = torch.load(f"weight/{model_choice}.pth", map_location=device)
                    model.load_my_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    print("Loaded pre-trained model with success.")
                except FileNotFoundError:
                    raise FileNotFoundError
            elif model_choice in ["gcn", "gcn_dc"]:
                model = GCN()
                try:
                    state_dict = torch.load(f"weight/{model_choice}.pth", map_location=device)
                    model.load_my_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    print("Loaded pre-trained model with success.")
                except FileNotFoundError:
                    raise FileNotFoundError

            representations = []
            embedded_smiles = []
            for batch in tqdm(dataloader):
                batch.to(device)
                batch_smiles, batch_representations = model(batch)
                representations.append(batch_representations)
                embedded_smiles.extend(batch_smiles)

        data = pd.read_csv(data_path)
        y = []
        for smiles in embedded_smiles:
            row = data[data['SMILES'] == smiles]
            lnp_class = row.iloc[0]['LNP Class']
            y.append(lnp_class)

        # for idx, smiles in enumerate(embedded_smiles):
        #     descs = torch.tensor(data.loc[data['SMILES'] == smiles, 'desc_ABC':].values.tolist()).to(device)
        #     representations[idx] = torch.cat((representations[idx], descs), dim=1)

        with open(f'data/result/{model_choice}_invalid_smiles.txt', 'w') as file:
            for smiles, value in embeddable.items():
                if not value:
                    file.write(smiles + '\n')

        return torch.cat(representations, dim=0), torch.Tensor(y).to(device)

    else:
        raise ValueError("no embeddable molecules")
        # return embeddable, None, None
        
def train_svc(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # pca = PCA(n_components=128)
    # X = pca.fit_transform(X_scaled)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # param_grid = {
    #     'C': [0.1, 1, 10, 100],  # Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Specifies the kernel type to be used in the algorithm.
    #     # 'degree': [2, 3, 4],  # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    #     'gamma': ["scale", "auto"],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    #     # 'coef0': [0.0, 0.1, 0.5],  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    #     # 'shrinking': [True, False],  # Whether to use the shrinking heuristic.
    #     # 'probability': [True, False],  # Whether to enable probability estimates.
    #     # 'tol': [1e-3, 1e-4],  # Tolerance for stopping criterion.
    #     # 'class_weight': [None, 'balanced'],  # Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.
    #     # 'decision_function_shape': ['ovo', 'ovr'],  # Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) or the original one-vs-one ('ovo') decision function.
    # }

    param_grid = {
        'C': [0.1, 1, 10],
        # Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
        'kernel': ['linear', 'rbf', 'poly'],  # Specifies the kernel type to be used in the algorithm.
        'gamma': ["scale", "auto"],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    }

    svc = SVC()
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=scoring, refit="accuracy", cv=kfold)

    grid_search.fit(X, y)
    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_svc.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'

            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]

            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")

    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{model_version}_{args.embedding}_svc.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)


def train_mlp(model_version, data_path, args):
    assert args.split == "NA"
    assert args.embedding in ["unimol_no_h", "unimol_all_h", "gin", "gcn", "gin_agile"]

    df = pd.read_csv(data_path)
    smiles_list = df['SMILES'].tolist()
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_embedding(data_path=data_path, model_choice=args.embedding, smiles_list=smiles_list, device=args.device)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_embedding(data_path=data_path, model_choice=args.embedding, smiles_list=smiles_list, device=args.device)
    elif args.embedding in ["gin", "gcn", "gin_dc", "gcn_dc", "gin_agile", "gcn_agile", "gin_agile_dc", "gcn_agile_dc"]:
        X, y = get_MolCLR_embedding(data_path=data_path, smiles_list=smiles_list, model_choice=args.embedding, device=args.device)
    else:
        raise ValueError("unknown embedding type")
    
    params = {
        'hidden_layer_sizes': [256, 128, 64, 32],
        'activation': nn.ReLU,
        'solver': 'adam'
    }

    val_metric = {}
    train_metric = {}



    kf = StratifiedKFold(n_splits=args.k_folds, random_state=42, shuffle=True)
    

    accuracy_val = []
    precision_val = []
    recall_val = []
    f1_val = []
    auroc_val = []
    tp_val = []
    fp_val = []
    tn_val = []
    fn_val = []
    
    accuracy_train = []
    precision_train = []
    recall_train = []
    f1_train = []
    auroc_train = []
    tp_train = []
    fp_train = []
    tn_train = []
    fn_train = []   
    
    accuracy_train_last = []
    precision_train_last = []
    recall_train_last = []
    f1_train_last = []
    auroc_train_last = []
    tp_train_last = []
    fp_train_last = []
    tn_train_last = []
    fn_train_last = []   


    for train_index, val_index in kf.split(X.cpu().numpy(), y.cpu().numpy()):
        model = MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"],
                            activation=params["activation"],
                            solver=params["solver"],
                            device=args.device).to(args.device)

        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        if args.resample:
            ros = RandomOverSampler()
            X_train_fold, y_train_fold = ros.fit_resample(X_train_fold.cpu().numpy(), y_train_fold.cpu().numpy())
            X_train_fold = torch.from_numpy(X_train_fold).to(args.device)
            y_train_fold = torch.from_numpy(y_train_fold).to(args.device)
            

        model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        accuracy, precision, recall, f1, auroc, tp, fp, tn, fn = model.evaluate(X_val_fold, y_val_fold, version="best")
        accuracy_val.append(accuracy)
        precision_val.append(precision)
        recall_val.append(recall)
        f1_val.append(f1)
        auroc_val.append(auroc)
        tp_val.append(tp)
        fp_val.append(fp)
        tn_val.append(tn)
        fn_val.append(fn)
        
        accuracy, precision, recall, f1, auroc, tp, fp, tn, fn = model.evaluate(X_train_fold, y_train_fold, version="best")
                    
        accuracy_train.append(accuracy)
        precision_train.append(precision)
        recall_train.append(recall)
        f1_train.append(f1)
        auroc_train.append(auroc)
        tp_train.append(tp)
        fp_train.append(fp)
        tn_train.append(tn)
        fn_train.append(fn)            
        
        accuracy, precision, recall, f1, auroc, tp, fp, tn, fn = model.evaluate(X_train_fold, y_train_fold, version="last")
        accuracy_train_last.append(accuracy)
        precision_train_last.append(precision)
        recall_train_last.append(recall)
        f1_train_last.append(f1)
        auroc_train_last.append(auroc)
        tp_train_last.append(tp)
        fp_train_last.append(fp)
        tn_train_last.append(tn)
        fn_train_last.append(fn)           
    
    if args.resample:
        surfix = "_resample"
    else:
        surfix = ""   

    with open(f"log/{model_version}_{args.embedding}_mlp{surfix}.log", "a") as file:
        file.write("cross validation metric:\n")
        file.write(f"accuracy: {np.mean(accuracy_val)}±{np.std(accuracy_val)}\n")
        file.write(f"precision: {np.mean(precision_val)}±{np.std(precision_val)}\n")
        file.write(f"recall: {np.mean(recall_val)}±{np.std(recall_val)}\n")
        file.write(f"f1: {np.mean(f1_val)}±{np.std(f1_val)}\n")
        file.write(f"auroc: {np.mean(auroc_val)}±{np.std(auroc_val)}\n")
        file.write(f"tp: {np.mean(tp_val)}±{np.std(tp_val)}\n")
        file.write(f"fp: {np.mean(fp_val)}±{np.std(fp_val)}\n")
        file.write(f"tn: {np.mean(tn_val)}±{np.std(tn_val)}\n")
        file.write(f"fn: {np.mean(fn_val)}±{np.std(fn_val)}\n")
        
        
        file.write("how does best validation models perform on training set:\n")
        file.write(f"accuracy: {np.mean(accuracy_train)}±{np.std(accuracy_train)}\n")
        file.write(f"precision: {np.mean(precision_train)}±{np.std(precision_train)}\n")
        file.write(f"recall: {np.mean(recall_train)}±{np.std(recall_train)}\n")
        file.write(f"f1: {np.mean(f1_train)}±{np.std(f1_train)}\n")
        file.write(f"auroc: {np.mean(auroc_train)}±{np.std(auroc_train)}\n")
        file.write(f"tp: {np.mean(tp_train)}±{np.std(tp_train)}\n")
        file.write(f"fp: {np.mean(fp_train)}±{np.std(fp_train)}\n")
        file.write(f"tn: {np.mean(tn_train)}±{np.std(tn_train)}\n")
        file.write(f"fn: {np.mean(fn_train)}±{np.std(fn_train)}\n")
        
        file.write("how does last-epoch models perform on training set:\n")
        file.write(f"accuracy: {np.mean(accuracy_train_last)}±{np.std(accuracy_train_last)}\n")
        file.write(f"precision: {np.mean(precision_train_last)}±{np.std(precision_train_last)}\n")
        file.write(f"recall: {np.mean(recall_train_last)}±{np.std(recall_train_last)}\n")
        file.write(f"f1: {np.mean(f1_train_last)}±{np.std(f1_train_last)}\n")
        file.write(f"auroc: {np.mean(auroc_train_last)}±{np.std(auroc_train_last)}\n")
        file.write(f"tp: {np.mean(tp_train_last)}±{np.std(tp_train_last)}\n")
        file.write(f"fp: {np.mean(fp_train_last)}±{np.std(fp_train_last)}\n")
        file.write(f"tn: {np.mean(tn_train_last)}±{np.std(tn_train_last)}\n")
        file.write(f"fn: {np.mean(fn_train_last)}±{np.std(fn_train_last)}\n")
    
    
def train_knn(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplementedError

    knn = KNeighborsClassifier()


    pipeline = Pipeline(steps=[('knn', knn)])

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']}

    param_grid = {f'knn__{key}': value for key, value in param_grid.items()}

    grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring=scoring, refit="accuracy", verbose=1)
    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    if args.resample:
        prefix = "resampled_"
    else:
        prefix = ""
    with open(f"log/{prefix}{model_version}_{args.embedding}_knn.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'

            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]

            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")

    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{prefix}{model_version}_{args.embedding}_knn.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)


def train_LogisticRegression(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {'C': np.logspace(-4, 4, 20), 'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2']}

    logreg = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring=scoring, refit="accuracy", cv=kfold)
    grid_search.fit(X_scaled, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_LogisticRegression.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'

            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]

            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")

    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{model_version}_{args.embedding}_LogisticRegression.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)


def train_RidgeClassifier(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {'alpha': np.logspace(-6, 6, 13)}

    ridge = RidgeClassifier()
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring=scoring, refit="accuracy", cv=kfold)
    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_RidgeClassifier.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'

            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]

            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")

    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{model_version}_{args.embedding}_RidgeClassifier.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)


def train_RandomForestClassifier(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # param_grid = {
    # 'n_estimators': [50, 100, 200],  # 树的数量
    # 'max_features': ['auto', 'sqrt', 'log2'],  # 在分裂节点时考虑的特征数量
    # 'max_depth': [None, 10, 20, 30],  # 树的最大深度
    # 'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
    # 'min_samples_leaf': [1, 2, 4],  # 在叶节点处需要的最小样本数
    # 'bootstrap': [True, False]  # 是否使用bootstrap采样
    # }

    param_grid = {
        'n_estimators': [50, 100],  # 树的数量
        'max_features': ['sqrt', 'log2'],  # 在分裂节点时考虑的特征数量
        'max_depth': [10, 20, 30],  # 树的最大深度
    }

    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=kfold, scoring=scoring,
                               refit="accuracy", n_jobs=-1)
    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_RandomForestClassifier.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'

            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]

            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")

    svc_best_model = grid_search.best_estimator_

    with open(f"weight/{model_version}_{args.embedding}_RandomForestClassifier.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)


def train_GradientBoostingClassifier(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # param_grid = {
    #     'n_estimators': [100, 200, 300],  # 建立的弱学习器的数量
    #     'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    #     'max_depth': [3, 4, 5],  # 每个决策树的最大深度
    #     'min_samples_split': [2, 3, 4],  # 分裂内部节点所需的最小样本数
    #     'min_samples_leaf': [1, 2, 3],  # 在叶节点处需要的最小样本数
    #     'max_features': [None, 'sqrt', 'log2'],  # 寻找最佳分割时要考虑的特征数量
    #     'subsample': [0.8, 0.9, 1.0]  # 用于拟合各个基础学习器的样本比例
    # }

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
    }

    gb_classifier = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=kfold, scoring=scoring,
                               refit="accuracy", n_jobs=-1)
    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_GradientBoostingClassifier.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'

            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]

            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")

    svc_best_model = grid_search.best_estimator_

    with open(f"weight/{model_version}_{args.embedding}_GradientBoostingClassifier.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)