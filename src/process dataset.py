import os
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from mordred import Calculator, descriptors

os.chdir("/root/autodl-tmp/lnp_class")

def process_data(raw_data_path):
    """
    remove duplicates and add train-test split
    :param raw_data_path:
    :return:
    """
    print(os.getcwd())
    # remove duplicates
    df = pd.read_csv(raw_data_path)
    duplicates = df[df.duplicated('SMILES', keep=False)]
    duplicates_dict = duplicates.groupby('SMILES').apply(lambda x: x.index.tolist()).to_dict()


    with open(f"data/result/{os.path.basename(raw_data_path).split('.')[0]}_duplicates.txt", 'w') as f:
        for smiles, indexes in duplicates_dict.items():
            f.write(f"Duplicated SMILES: {smiles}, Indexes: {indexes}\n")

    df = df.drop_duplicates('SMILES', keep=False)

    # add train test split
    # add train test split for Raw data
    raw_df = df[df['category'] == 'Raw']
    assert raw_df['LNP Class'].nunique() == 2

    class_0 = raw_df[raw_df['LNP Class'] == 0]
    class_1 = raw_df[raw_df['LNP Class'] == 1]

    train_0, test_0 = train_test_split(class_0, test_size=0.2, random_state=42)
    train_1, test_1 = train_test_split(class_1, test_size=0.2, random_state=42)

    train_df = pd.concat([train_0, train_1])
    test_df = pd.concat([test_0, test_1])

    df['split'] = 'NA'
    df.loc[train_df.index, 'split'] = 'train'
    df.loc[test_df.index, 'split'] = 'test'

    df.to_csv(f"data/raw/{os.path.basename(raw_data_path).split('.')[0]}_no_duplicates.csv", index=False)

def merge_modred_descriptor(data_path):
    """
    generated modred descriptor
    :param data_path:
    :return:
    """
    print(os.getcwd())
    df = pd.read_csv(data_path)
    smiles_list = df["SMILES"].tolist()
    valid_smiles_list = []
    invalid_smiles_list = []
    valid_mols_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles_list.append(smiles)
            valid_mols_list.append(mol)
        else:
            invalid_smiles_list.append(smiles)

    calc = Calculator(descriptors, ignore_3D=True)
    desc_df = calc.pandas(valid_mols_list)
    desc_df.insert(0, "SMILES", valid_smiles_list)
    desc_df.columns = [desc_df.columns[0]] + ['desc_' + col for col in desc_df.columns[1:]]

    desc_df.to_csv(f"data/raw/{os.path.basename(data_path).split('.')[0]}_desc.csv", index=False)

    for col in desc_df.columns[1:]:
        if desc_df[col].apply(lambda x: isinstance(x, str)).any():
            desc_df.drop(col, axis=1, inplace=True)

    merged_df = pd.merge(df, desc_df, on='SMILES', how='inner')
    merged_df.to_csv(data_path, index=False)

    with open(f"data/result/{os.path.basename(data_path).split('.')[0]}_invalid_smiles.txt", "w") as file:
        for invalid_smiles in invalid_smiles_list:
            file.write(f"{invalid_smiles}\n")

def remove_str_col(data_path):
    df = pd.read_csv(data_path)
    for column in df.columns[1:]:
        print(df[column].dtype)
        if df[column].dtype == object:
            df = df.drop(column, axis=1)

    return df

    # df.to_csv(data_path)

def merge(original_data_path, desc_data_path):
    original_df = pd.read_csv(original_data_path)
    desc_df = pd.read_csv(desc_data_path)
    new_df = pd.merge(original_df, desc_df, on='SMILES', how='inner')
    for column in new_df.columns[6:]:
        if new_df[column].dtype == object:
            new_df = new_df.drop(column, axis=1)

    print(new_df)
    new_df.to_csv("data/raw/lnp_classification_no_duplicates.csv")


if __name__ == "__main__":
    print(os.getcwd())
    raw_data_path = "data/raw/lnp_classification.csv"
    data_path_no_duplicates = "data/raw/lnp_classification_no_duplicates.csv"

    # process_data(raw_data_path) # remove duplicates and split train-test
    # merge_modred_descriptor(data_path_no_duplicates) # generate modred descriptor and merged
    # remove_str_col(data_path_no_duplicates)
    merge("data/raw/lnp_classification_no_duplicates.csv", "data/raw/lnp_classification_no_duplicates_desc.csv")










