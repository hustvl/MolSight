from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog('rdApp.*')
import os
import random
import pandas as pd
from tqdm import tqdm

def apply_random_style(drawer):
    """
    随机从预设风格中选择一个，并应用到drawer.drawOptions()上。
    经典风格概率最高，其他为轻微变化。
    """
    styles = [
        "classic", "slight_blue_bg", "light_gray_bg", "with_atom_indices",
        "thicker_lines",
    ]
    # 经典风格权重高，轻微变化占少数
    style = random.choices(styles, weights=[60, 10, 10, 10, 10], k=1)[0]

    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.useBWAtomPalette()  # 强制黑白绘图

    if style == "classic":
        pass  # 保持默认
    elif style == "slight_blue_bg":
        opts.setBackgroundColour((0.95, 0.95, 1))
    elif style == "light_gray_bg":
        opts.setBackgroundColour((0.95, 0.95, 0.95))
    elif style == "with_atom_indices":
        opts.addAtomIndices = True
    elif style == "thicker_lines":
        opts.bondLineWidth = 2.0

    return style

def main():
    os.makedirs("data/stereo/train", exist_ok=True)
    smiles_file = "stereo_smiles.csv"
    data_df = pd.read_csv(smiles_file)

    for idx in tqdm(range(42095, len(data_df))):
        row = data_df.iloc[idx]
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid idx {idx} SMILES: {smiles}")
            continue

        drawer = rdMolDraw2D.MolDraw2DCairo(512, 512)
        style = apply_random_style(drawer)

        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()

        img_path = f"data/stereo/train/{idx}.png"
        with open(img_path, "wb") as f:
            f.write(drawer.GetDrawingText())

if __name__ == "__main__":
    main()
