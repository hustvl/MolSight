import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import DataStructs

def tanimoto_reward_fn(pred_smiles_list, gt_smiles_list):
    rewards = []
    for pred, gt in zip(pred_smiles_list, gt_smiles_list):
        try:
            mol_pred = Chem.MolFromSmiles(pred)
            mol_gt = Chem.MolFromSmiles(gt)
            if mol_pred is None or mol_gt is None:
                rewards.append(0.0)
                continue
            fp_pred = Chem.RDKFingerprint(mol_pred)
            fp_gt = Chem.RDKFingerprint(mol_gt)
            tanimoto_sim = DataStructs.FingerprintSimilarity(fp_pred, fp_gt)
            rewards.append(tanimoto_sim)
        except:
            rewards.append(0.0)
    return torch.tensor(rewards, dtype=torch.float32) * 0.4

def stereochem_reward_fn(pred_list, gt_list):
    rewards = []
    for pred, gt in zip(pred_list, gt_list):
        try:
            mol_pred = Chem.MolFromSmiles(pred)
            mol_gt = Chem.MolFromSmiles(gt)
            if mol_pred is None or mol_gt is None:
                rewards.append(0.0)
                continue

            # 完全结构相同（包括手性/构型）
            if Chem.MolToInchiKey(mol_pred) == Chem.MolToInchiKey(mol_gt):
                rewards.append(1.0)
            # 错误的异构体
            #elif Chem.MolToInchi(mol_pred, options='/FixedH') == Chem.MolToInchi(mol_gt, options='/FixedH'):
            #    rewards.append(0.5)
            # 结构基本相同
            elif mol_pred.GetNumAtoms() == mol_gt.GetNumAtoms():
                rewards.append(0.3)
            else:
                rewards.append(0.1)
        except:
            rewards.append(0.0)
    return torch.tensor(rewards, dtype=torch.float32) * 0.6
