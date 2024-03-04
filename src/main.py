import pandas as pd
import sys
import pickle as pkl
import os
from tqdm import tqdm
from threading import Thread, Lock
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch.utils.data import DataLoader
import argparse
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

from MolCLR_dataset import MolTestDatasetWrapper, MolTestDatasetWrapper_smiles
from gcn import GCN
from ginet import GINet
from training import train_svc, train_mlp, train_knn, train_LogisticRegression, train_RidgeClassifier, \
    train_RandomForestClassifier, train_GradientBoostingClassifier, get_UniMol_embedding, get_MolCLR_embedding
import random
import numpy as np
from mixed_training import train_mlp_from_mixed_dataset
import pickle as pkl
import math

os.chdir("/home/feiyanlin/projects/lnp class")
def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



def calculate_3D_structure(raw_data_path):
    def get_smiles_list_():
        data_df = pd.read_csv(raw_data_path)
        smiles_list = data_df["SMILES"].tolist()
        smiles_list = list(set(smiles_list))
        print(len(smiles_list))
        return smiles_list

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
                with open(f'data/result/{os.path.basename(raw_data_path).split(".")[0]}_invalid_smiles.txt', 'a') as f:
                    f.write('MolFromSmiles error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                with open(f'data/result/{os.path.basename(raw_data_path).split(".")[0]}_invalid_smiles.txt', 'a') as f:
                    f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                with open(f'data/result/{os.path.basename(raw_data_path).split(".")[0]}_invalid_smiles.txt', 'a') as f:
                    f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
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
    smiles_list = get_smiles_list_()
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
    pkl.dump(smiles_to_conformation_dict,
             open(f'data/intermediate/{os.path.basename(raw_data_path).split(".")[0]}_smiles_to_conformation_dict.pkl',
                  'wb'))
    print('Valid smiles count:', len(smiles_to_conformation_dict))



def user_single_smiles_predict(smiles: str, device, embedding: dict, trained_model: dict):
    """
    get embedding, load trained_model, return prediction (direct & rule-based prediction of accum)
    e.g., embedding = {"Formal charge": "gin", "Q_vsa_Ppos": "gin", "vsa_don": "gin", "PA Accum": "gin"}
    trained_model = {"Formal charge": "knn", "Q_vsa_Ppos": "knn", "vsa_don": "knn", "PA Accum": "knn"}
    """
    assert (i in embedding for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    assert (i in trained_model for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    X = {}
    embeddable = True

    for label, embed_type in embedding.items():
        if embed_type == "gin":
            embeddable, _, X[label] = get_MolCLR_embedding(smiles_list=[smiles], model_choice="gin", device=device)
            if embeddable[smiles] == False:
                return None, None, None, None, None, None
        elif embed_type == "gcn":
            embeddable, _, X[label] = get_MolCLR_embedding(smiles_list=[smiles], model_choice="gcn", device=device)
            if embeddable[smiles] == False:
                return None, None, None, None, None, None
        elif embed_type == "unimol_no_h":
            _, X[label] = get_UniMol_embedding(remove_hydrogen=True, smiles_list=[smiles], device=device)
        elif embed_type == "unimol_all_h":
            _, X[label] = get_UniMol_embedding(remove_hydrogen=False, smiles_list=[smiles], device=device)
        elif embedding == "mordred":
            raise NotImplementedError

    with open(f'weight/Formal charge Class_{embedding["Formal charge"]}_{trained_model["Formal charge"]}.pkl',
              'rb') as Formal_charge_model_file:
        Formal_charge_model = pkl.load(Formal_charge_model_file)
    with open(f'weight/Q_vsa_Ppos Class_{embedding["Q_vsa_Ppos"]}_{trained_model["Q_vsa_Ppos"]}.pkl',
              'rb') as Q_vsa_Ppos_model_file:
        Q_vsa_Ppos_model = pkl.load(Q_vsa_Ppos_model_file)
    with open(f'weight/vsa_don Class_{embedding["vsa_don"]}_{trained_model["vsa_don"]}.pkl',
              'rb') as vsa_don_model_file:
        vsa_don_model = pkl.load(vsa_don_model_file)
    with open(f'weight/PA Accum Class_{embedding["PA Accum"]}_{trained_model["PA Accum"]}.pkl',
              'rb') as PA_Accum_model_file:
        PA_Accum_model = pkl.load(PA_Accum_model_file)

    Formal_charge_prediction = Formal_charge_model.predict(X["Formal charge"].cpu().detach().numpy()).item()
    Q_vsa_Ppos_prediction = Q_vsa_Ppos_model.predict(X["Q_vsa_Ppos"].cpu().detach().numpy()).item()
    vsa_don_prediction = vsa_don_model.predict(X["vsa_don"].cpu().detach().numpy()).item()
    rule_PA_Accum_prediction = Formal_charge_prediction and (Q_vsa_Ppos_prediction or vsa_don_prediction)
    direct_PA_Accum_prediction = PA_Accum_model.predict(X["PA Accum"].cpu().detach().numpy()).item()
    return Formal_charge_prediction, Q_vsa_Ppos_prediction, vsa_don_prediction, rule_PA_Accum_prediction, direct_PA_Accum_prediction



def user_smiles_list_predict(smiles_list_path: str, device, embedding: dict, trained_model: dict):
    Formal_charge_prediction = {}
    Q_vsa_Ppos_prediction = {}
    vsa_don_prediction = {}
    rule_PA_Accum_prediction = {}
    direct_PA_Accum_prediction = {}

    df = pd.read_csv(smiles_list_path)

    if "SMILES" in df.columns:
        smiles_list = df["SMILES"].tolist()
    else:
        print("The CSV file does not contain a 'smiles' column.")
        sys.exit()

    assert (i in embedding for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    assert (i in trained_model for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    X = {}

    for label, embed_type in embedding.items():
        if embed_type == "gin":
            embeddable, _, X[label] = get_MolCLR_embedding(smiles_list=smiles_list, model_choice="gin", device=device)
        elif embed_type == "gcn":
            embeddable, _, X[label] = get_MolCLR_embedding(smiles_list=smiles_list, model_choice="gcn", device=device)
        elif embed_type == "unimol_no_h":
            _, X[label] = get_UniMol_embedding(remove_hydrogen=True, smiles_list=smiles_list, device=device)
        elif embed_type == "unimol_all_h":
            _, X[label] = get_UniMol_embedding(remove_hydrogen=False, smiles_list=smiles_list, device=device)
        elif embedding == "mordred":
            raise NotImplementedError

    with open(f'weight/Formal charge Class_{embedding["Formal charge"]}_{trained_model["Formal charge"]}.pkl',
              'rb') as Formal_charge_model_file:
        Formal_charge_model = pkl.load(Formal_charge_model_file)
    with open(f'weight/Q_vsa_Ppos Class_{embedding["Q_vsa_Ppos"]}_{trained_model["Q_vsa_Ppos"]}.pkl',
              'rb') as Q_vsa_Ppos_model_file:
        Q_vsa_Ppos_model = pkl.load(Q_vsa_Ppos_model_file)
    with open(f'weight/vsa_don Class_{embedding["vsa_don"]}_{trained_model["vsa_don"]}.pkl',
              'rb') as vsa_don_model_file:
        vsa_don_model = pkl.load(vsa_don_model_file)
    with open(f'weight/PA Accum Class_{embedding["PA Accum"]}_{trained_model["PA Accum"]}.pkl',
              'rb') as PA_Accum_model_file:
        PA_Accum_model = pkl.load(PA_Accum_model_file)

    Formal_charge_prediction = Formal_charge_model.predict(X["Formal charge"].cpu().detach().numpy())
    Q_vsa_Ppos_prediction = Q_vsa_Ppos_model.predict(X["Q_vsa_Ppos"].cpu().detach().numpy())
    vsa_don_prediction = vsa_don_model.predict(X["vsa_don"].cpu().detach().numpy())
    rule_PA_Accum_prediction = Formal_charge_prediction & (Q_vsa_Ppos_prediction | vsa_don_prediction)
    direct_PA_Accum_prediction = PA_Accum_model.predict(X["PA Accum"].cpu().detach().numpy())

    return smiles_list, Formal_charge_prediction, Q_vsa_Ppos_prediction, vsa_don_prediction, rule_PA_Accum_prediction, direct_PA_Accum_prediction



def get_dataloader_from_dataset(data_path, device, args):
    df = pd.read_csv(data_path)
    train_df = df.sample(n=args.train_ratio, random_state=42)
    test_df = df[~df.index.isin(train_df)]
    
    train_smiles_list = train_df["SMILES"].tolist()
    test_smiles_list = test_df["SMILES"].tolist()
    
    if args.embedding in ["gin", "gcn", "gin_dc", "gcn_dc", "gin_agile", "gcn_agile", "gin_agile_dc", "gcn_agile_dc"]:
        X_train, y_train = get_MolCLR_embedding(data_path=data_path, smiles_list=train_smiles_list, model_choice=args.embedding, device=device)
        X_test, y_test = get_MolCLR_embedding(data_path=data_path, smiles_list=test_smiles_list, model_choice=args.embedding, device=device)
    
    return X_train, y_train, X_test, y_test



def get_dataloader_from_mixed_dataset(data_path, device, args):

    df = pd.read_csv(data_path)

    primary_df = df[df['category'] == args.primary_category]
    related_df = df[df['category'] == args.related_category]
    related_train_df = related_df.sample(n=args.mix_number, random_state=42)
    related_test_df = related_df[~related_df.index.isin(related_train_df.index)]
                         
    primary_smiles_list = primary_df['SMILES'].tolist()
    train_related_smiles_list = related_train_df['SMILES'].tolist()
    test_related_smiles_list = related_test_df['SMILES'].tolist()
    if args.embedding in ["unimol_no_h", "unimol_all_h"]:
        X_primary, y_primary = get_UniMol_embedding(data_path=data_path, model_choice=args.embedding, smiles_list=primary_smiles_list, device=device)
        X_train_related, y_train_related = get_UniMol_embedding(data_path=data_path, model_choice=args.embedding, smiles_list=train_related_smiles_list, device=device)
        X_test_related, y_test_related = get_UniMol_embedding(data_path=data_path, model_choice=args.embedding, smiles_list=test_related_smiles_list, device=device)
    elif args.embedding in ["gin", "gcn", "gin_dc", "gcn_dc", "gin_agile", "gcn_agile", "gin_agile_dc", "gcn_agile_dc"]:
        X_primary, y_primary = get_MolCLR_embedding(data_path=data_path, smiles_list=primary_smiles_list, model_choice=args.embedding, device=device)
        X_train_related, y_train_related = get_MolCLR_embedding(data_path=data_path, smiles_list=train_related_smiles_list, model_choice=args.embedding, device=device)
        X_test_related, y_test_related = get_MolCLR_embedding(data_path=data_path, smiles_list=test_related_smiles_list, model_choice=args.embedding, device=device)
    else:
        raise ValueError("unknown embedding type")

    return X_primary, y_primary, X_train_related, y_train_related, X_test_related, y_test_related

# def train(data_path:str, args):
#     X_train, y_train, X_test, y_test = get_dataloader_from_dataset(data_path, device=args.device, args=args)
#     if args.classifier == "mlp":
#         train_mlp_from_dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, args=args)

def mixed_train(data_path:str, args):
    """
    需要传入的参数包括primary category, related category, data_path, mix_ratio, args里和model有关的参数
    :return:
    """
    X_primary, y_primary, X_train_related, y_train_related, X_test_related, y_test_related = get_dataloader_from_mixed_dataset(data_path=data_path, 
                                                                         device=args.device, args=args)

    if args.classifier == "mlp":

        train_mlp_from_mixed_dataset(X_primary=X_primary, y_primary=y_primary, X_train_related=X_train_related,
                                        y_train_related=y_train_related, X_test_related=X_test_related,
                                        y_test_related=y_test_related, k_folds=args.k_folds, args=args)
        

        # with open(f"log/{args.embedding}_{args.classifier}_{args.mix_ratio}Mix_{args.k_folds}Fold{surfix}.txt", "w") as file:
        #     file.write("best cross validation performance:\n")
        #     for key, value in best_val_metric.items():
        #         file.write(f"{key}: {value}\n")
        #     file.write("performance on related test set:\n")
        #     for key, value in test_related_metric.items():
        #         file.write(f"{key}: {value}\n")
        #     file.write("how does this best valid model perform on train set:\n")
        #     for key, value in best_train_fitting_metric.items():
        #         file.write(f"{key}: {value}\n")
        #     file.write("how does the last model perform on train set:\n")
        #     for key, value in train_fitting_metric_last.items():
        #         file.write(f"{key}: {value}\n")



def main4mixed_train():

    ## 把KNN, random forest, gradient boosting也实现一下

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/lnp_classification_no_duplicates.csv", type=str)
    parser.add_argument("--label", default="LNP Class", type=str, help="label")
    parser.add_argument("--primary_category", default="Raw", type=str)
    parser.add_argument("--related_category", default="DC2.4", type=str)
    parser.add_argument("--mix_number", default=10, type=int)
    parser.add_argument("--gpu", default="cuda:3", type=str)
    parser.add_argument("--embedding", default="gin", type=str,
                        choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "gin_agile", "gin_agile_dc",
                                 "gin_dc", "gcn_dc"])
    parser.add_argument("--classifier", default="mlp", type=str,
                        choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier",
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--k_folds", default=3, type=int)
    parser.add_argument("--k", default=20, type=int, help="top_k_recall")
    parser.add_argument("--resample", default=False, type=bool)
    parser.add_argument("--neptune", default=True, type=bool)

    args = parser.parse_args()
    set_random_seed(1024)
    if torch.cuda.is_available() and args.gpu != "cpu":
        args.device = args.gpu
        torch.cuda.set_device(args.device)
    else:
        args.device = "cpu"
        
    
    if args.neptune:
        import neptune
        args.run = neptune.init_run(project="feiyl/lnp-class",
                                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YTkzM2EzYS1mZTIwLTQ4ZjAtYTc0NS01YjZlMzk2ZDQ2ODQifQ==",)
        args.run["sys/tags"].add(args.embedding)
    else:
        args.run = None

    mixed_train(data_path=args.data_path, args=args)

    if args.run is not None:
        args.run.stop()



def get_predicted_properties(smiles):
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_single_smiles_prediction", default=True, type=bool)
    # parser.add_argument("--user_smiles_list_prediction", default=False, type=bool)
    # parser.add_argument("--user_smiles_list", default="user_data/user_smiles_list.csv", type=str)
    parser.add_argument("--gpu", default="cuda:4", type=str)
    parser.add_argument("--user_single_smiles", type=str)
    parser.add_argument("--Formal_charge_embedding", default="gcn", type=str,
                        choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--Q_vsa_Ppos_embedding", default="gin", type=str,
                        choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--vsa_don_embedding", default="gin", type=str,
                        choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--PA_Accum_embedding", default="gcn", type=str,
                        choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--Formal_charge_model", default="mlp", type=str,
                        choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier",
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--Q_vsa_Ppos_model", default="mlp", type=str,
                        choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier",
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--vsa_don_model", default="mlp", type=str,
                        choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier",
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--PA_Accum_model", default="mlp", type=str,
                        choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier",
                                 "RandomForestClassifier", "GradientBoostingClassifier"])

    args = parser.parse_args()
    set_random_seed(1024)
    if torch.cuda.is_available() and args.gpu != "cpu":
        args.device = args.gpu
        torch.cuda.set_device(args.device)
    else:
        args.device = "cpu"

    args.user_single_smiles = smiles
    embedding = {"Formal charge": args.Formal_charge_embedding, "Q_vsa_Ppos": args.Q_vsa_Ppos_embedding,
                 "vsa_don": args.vsa_don_embedding, "PA Accum": args.PA_Accum_embedding}
    trained_model = {"Formal charge": args.Formal_charge_model, "Q_vsa_Ppos": args.Q_vsa_Ppos_model,
                     "vsa_don": args.vsa_don_model, "PA Accum": args.PA_Accum_model}
    (Formal_charge_prediction, Q_vsa_Ppos_prediction, vsa_don_prediction,
     rule_PA_Accum_prediction, direct_PA_Accum_prediction) = user_single_smiles_predict(smiles=args.user_single_smiles,
                                                                                        device=args.device,
                                                                                        embedding=embedding,
                                                                                        trained_model=trained_model)

    if Formal_charge_prediction is None:
        return {
            "Formal Charge": None,
            "Q_vsa_Ppos": None,
            "vsa_don": None,
            "PA Accumulation": None
        }

    else:
        if Formal_charge_prediction:
            Foraml_charge_prompt = ">= 0.98"
        else:
            Foraml_charge_prompt = "< 0.98"
        if Q_vsa_Ppos_prediction:
            Q_vsa_Ppos_prompt = ">= 80"
        else:
            Q_vsa_Ppos_prompt = "< 80"
        if vsa_don_prediction:
            vsa_don_prompt = ">= 23"
        else:
            vsa_don_prompt = "< 23"
        if rule_PA_Accum_prediction:
            print(
                "rule-base prediction of PA Accumulation [(vsa_don >= 23) && ((Q_vsa_Ppos >= 80) || (Formal Charge >= 0.98))]: Yes")
        else:
            print(
                "rule-base prediction of PA Accumulation [(vsa_don >= 23) && ((Q_vsa_Ppos >= 80) || (Formal Charge >= 0.98))]: No")
        if direct_PA_Accum_prediction:
            direct_PA_Accum_prompt = "Accumalation: Yes"
        else:
            direct_PA_Accum_prompt = "Accumalation: No"

        return {
            "Formal Charge": Formal_charge_prediction,
            "Q_vsa_Ppos": Q_vsa_Ppos_prediction,
            "vsa_don": vsa_don_prediction,
            "PA Accumulation": direct_PA_Accum_prediction
        }

def main4single_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/lnp_classification_no_duplicates_Raw.csv", type=str)
    parser.add_argument("--split", default="NA", type=str)
    parser.add_argument("--gpu", default="cuda:4", type=str)
    parser.add_argument("--embedding", default="gin", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "gin_agile", "mordred"])
    parser.add_argument("--classifier", default="mlp", type=str, 
                        choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--resample", default=True, type=bool)
    parser.add_argument("--model_version", default="Raw", type=str)
    parser.add_argument("--k_folds", default=5, type=int)
    
    args = parser.parse_args()
    set_random_seed(1024)
    if torch.cuda.is_available() and args.gpu != "cpu":
        args.device = args.gpu
        torch.cuda.set_device(args.device)
    else:
        args.device = "cpu"
        
    if args.classifier == "svc":
        train_svc(model_version=args.model_version, data_path=args.data_path, args=args) # 0.698, with h; accum: 0.665 (unimol), 0.668 (gin)
    elif args.classifier == "mlp":
        train_mlp(model_version=args.model_version, data_path=args.data_path, args=args) # accum: 0.58 (unimol), 0.68 (gins), ★0.70 (gcn) 
    elif args.classifier == "knn":
        train_knn(model_version=args.model_version, data_path=args.data_path, args=args) # # 0.694; accum: 0.669 (unimol), 0.659 (gin)， ★0.701 (gcn)
    elif args.classifier == "LogisticRegression":
        train_LogisticRegression(model_version=args.model_version, data_path=args.data_path, args=args)
    elif args.classifier == "RidgeClassifier":
        train_RidgeClassifier(model_version=args.model_version, data_path=args.data_path, args=args)
    elif args.classifier == "RandomForestClassifier":
        train_RandomForestClassifier(model_version=args.model_version, data_path=args.data_path, args=args)
    elif args.classifier == "GradientBoostingClassifier":
        train_GradientBoostingClassifier(model_version=args.model_version, data_path=args.data_path, args=args)
    print("training: All is well!")

if __name__ == "__main__":
    main4mixed_train()


