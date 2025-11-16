import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import cv2
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle
import lmdb

from molsight.indigo import Indigo
from molsight.indigo.renderer import IndigoRenderer
from molsight.augment import CropWhite

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Scaffold Split ----------
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def scaffold_split(df, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42):
    random.seed(seed)
    scaffolds = defaultdict(list)

    for i, smi in enumerate(df['smiles']):
        scaffold = get_scaffold(smi)
        if not scaffold:
            # 给空骨架一个唯一标识（防止全聚在一个组）
            scaffold = f"_no_scaffold_{i}"
        scaffolds[scaffold].append(i)

    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)

    total = len(df)
    n_train = int(frac_train * total)
    n_valid = int(frac_valid * total)

    train_idx, valid_idx, test_idx = [], [], []

    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train:
            train_idx += group
        elif len(valid_idx) + len(group) <= n_valid:
            valid_idx += group
        else:
            test_idx += group

    # 若验证集或测试集太小，可进行补充
    leftovers = list(set(range(total)) - set(train_idx) - set(valid_idx) - set(test_idx))
    for idx in leftovers:
        if len(valid_idx) < n_valid:
            valid_idx.append(idx)
        else:
            test_idx.append(idx)

    return df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]

def load_lmdb(lmdb_path, test_or_valid=False):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    smiles_lst = []
    target_lst = []
    coordinates_lst = []
    atoms_lst = []
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        smiles_lst.append(str(data['smi']))
        target_lst.append(np.array(data['target']))
        if test_or_valid:
            coordinates_lst.append(np.array([data['coordinates'][0]]))
        else:
            coordinates_lst.append(np.array(data['coordinates']))
        atoms_lst.append(data['atoms'])
    

    return [smiles_lst, np.array(target_lst), coordinates_lst, atoms_lst]

# ---------- Dataset ----------
class MoleculeDataset(Dataset):
    def __init__(self, smiles, labels, features):
        self.smiles = smiles
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ---------- MLP Model ----------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

# ---------- Main ----------
def main():
    seed = 0
    print(seed)
    set_seed(seed)
    # 1. Load CSV
    #df = pd.read_csv("MoleculeNet/tox21.csv")  # 应含有列 'smiles' 和一个或多个标签列（如 NR-AR, etc.）
    #task_cols = df.columns.drop(['mol_id', 'smiles'])
    #df = df.dropna()  # 只保留有完整标签的行

    dataset_path = 'molecular_property_prediction'
    dataset_name = 'tox21'
    # 0: smiles, 1: labels, bbbp 2 classes, 1369 vs 262, 2: 3D position, 11 conformations, 11 * 50 * 3, 3: atom types
    train_smiles, y_train, _, _ = load_lmdb(os.path.join(dataset_path, dataset_name, "train.lmdb"))
    valid_smiles, y_val, _, _ = load_lmdb(os.path.join(dataset_path, dataset_name, "valid.lmdb"))
    test_smiles, y_test, _, _ = load_lmdb(os.path.join(dataset_path,dataset_name, "test.lmdb"))

    # 2. Featurization using EfficientViT
    image_encoder = timm.create_model('efficientvit_l1', pretrained=False)
    # remove head of the image encoder
    image_encoder.head = nn.Identity()
    scale_factors = [1, 2, 4]
    num_channels = [int(ratio * image_encoder.num_features) for ratio in [0.25, 0.5, 1]]
    image_encoder.fuse_blocks = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
        )
        for dim, scale in zip(num_channels, scale_factors)
    ])
    image_encoder.point_conv = nn.Sequential(
        nn.Conv2d(sum(num_channels), 512, kernel_size=1, bias=False),
        nn.BatchNorm2d(512),
        nn.GELU(),
    )

    # load checkpoint
    ckpt_path = 'runs/pubchem_uspto_smiles_edges_aug_30/ckpt_model/epoch_26.pth'
    print("=> loading checkpoint from {}".format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_checkpoint = checkpoint["model"]
    model_checkpoint = {
        (k[21:] if k.startswith('module.') else k[14:]): v
        for k, v in model_checkpoint.items() if 'image_encoder' in k
    }
    missing_keys, unexpected_keys = image_encoder.load_state_dict(model_checkpoint, strict=False)
    print(f'Missing keys: {missing_keys}')
    print(f'Unexpected keys: {unexpected_keys}')
    image_encoder.eval()
    image_encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    pooling = nn.AdaptiveAvgPool2d((1, 1))

    # Image transformation
    input_size = 512
    trans_list = []
    trans_list.append(CropWhite(pad=5))
    trans_list.append(A.LongestMaxSize(max_size=input_size))
    trans_list.append(A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255)))
    trans_list.append(A.ToGray(p=1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trans_list += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    transform = A.Compose(trans_list)

    def featurize(smi):
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        indigo.setOption('render-output-format', 'png')
        indigo.setOption('render-background-color', '1,1,1')
        indigo.setOption('render-stereo-style', 'none')
        indigo.setOption('render-label-mode', 'hetero')
        indigo.setOption('render-font-family', 'Arial')

        try:
            mol = indigo.loadMolecule(smi)

            buf = renderer.renderToBuffer(mol)
            image = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
            success = True
        except Exception:
            image = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
            success = False
            print(f"Failed to render molecule: {smi}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        image = transform(image=image)['image']  # [C, H, W]
        image = image[None, :].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # [1, C, H, W]

        intermediates = image_encoder.forward_intermediates(image, indices=len(scale_factors), intermediates_only=True)
        feats = []
        for inter_feat, fuse_block in zip(intermediates, image_encoder.fuse_blocks):
            feats.append(fuse_block(inter_feat))
        # sum up the features
        image_features = torch.cat(feats, dim=1)  # [B, C, H, W]
        image_features = image_encoder.point_conv(image_features)
        embed = pooling(image_features).squeeze(-1).squeeze(-1)  # [B, C]
        
        return embed[0].detach().cpu().numpy()

    train_feature_list = [featurize(smi) for smi in tqdm(train_smiles, desc="Featurizing train set")]
    valid_feature_list = [featurize(smi) for smi in tqdm(valid_smiles, desc="Featurizing valid set")]
    test_feature_list = [featurize(smi) for smi in tqdm(test_smiles, desc="Featurizing test set")]
    #df['features'] = df['smiles'].apply(featurize)
    #df = df[df['features'].notnull()]  # 去除失败项

    # 3. Scaffold Split
    #train_df, val_df, test_df = scaffold_split(df, seed=seed)
    #print(f"Total: {len(df)}, Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    #if len(val_df) == 0:
    #    val_df = train_df.sample(frac=0.1, random_state=seed)

    # 4. 准备特征和标签
    scaler = StandardScaler()
    X_train = np.stack(train_feature_list)
    X_val = np.stack(valid_feature_list)
    X_test = np.stack(test_feature_list)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"unique labels in y_train: {np.unique(y_train)}")

    # 5. 构建数据集
    train_set = MoleculeDataset(train_feature_list, y_train, X_train)
    val_set = MoleculeDataset(valid_feature_list, y_val, X_val)
    test_set = MoleculeDataset(test_feature_list, y_test, X_test)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    # 6. 初始化模型
    input_dim = 512
    output_dim = y_train.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=512, output_dim=output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #criterion = nn.BCEWithLogitsLoss()
    def masked_bce_loss(logits, labels):
        mask = (labels != -1).float()
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        raw_loss = loss_fn(logits, torch.clamp(labels, min=0.0))
        loss = raw_loss * mask
        return loss.sum() / mask.sum()

    # 7. 训练
    best_valid_score = 0.0
    best_test_score = 0.0
    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = masked_bce_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f}")

        # 8. 评估
        def evaluate(loader):
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    logits = model(x)
                    prob = torch.sigmoid(logits).cpu().numpy()
                    y_pred.append(prob)
                    y_true.append(y.numpy())

            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
            aucs = []
            for i in range(y_true.shape[1]):
                try:
                    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    aucs.append(auc)
                except:
                    continue
            return np.nanmean(aucs), aucs

        val_auc, val_task_aux = evaluate(val_loader)
        test_auc, task_aucs = evaluate(test_loader)
        print(f"\nValidation ROC-AUC: {val_auc:.4f}")
        print(f"validation Per-task AUCs: {val_task_aux}")
        print(f"Test ROC-AUC: {test_auc:.4f}")
        print(f"Per-task AUCs: {task_aucs}")
        if val_auc > best_valid_score:
            best_valid_score = val_auc
            best_test_score = test_auc
        
    print(f"Best Valid ROC-AUC: {best_valid_score:.4f}, Best Test ROC-AUC: {best_test_score:.4f}\n")

if __name__ == "__main__":
    main()
