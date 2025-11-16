import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer

from molsight.chemistry import _postprocess_smiles

#ckpt_path = 'molscribe_model/swin_base_char_aux_1m.pth'

#model = MolScribe(ckpt_path, device=torch.device('cpu'))
#output = model.predict_image_file('assets/82a7978ed0c7be7216dc7764ba2585150cd8fd66e06ea72bffec23989cc71ca5.jpg', return_atoms_bonds=True, return_confidence=True)
#SMILES = output['smiles']
SMILES = 'O=C1O[C@@]2(CN3CCC2CC3)CN1C1=CC(Br)=CO1'
SMILES, _, success = _postprocess_smiles(SMILES)

replace_rgroup = False
if replace_rgroup:
    tokens = atomwise_tokenizer(SMILES)
    for j, token in enumerate(tokens):
        if token[0] == '[' and token[-1] == ']':
            symbol = token[1:-1]
            if symbol[0] == 'R' and symbol[1:].isdigit():
                tokens[j] = f'[{symbol[1:]}*]'
            elif Chem.AtomFromSmiles(token) is None:
                tokens[j] = '*'
    SMILES = ''.join(tokens)

SMILES = Chem.CanonSmiles(SMILES)
print(SMILES)
mol = Chem.MolFromSmiles(SMILES)

'''atoms = [atom for atom in mol.GetAtoms()]
for atom in atoms:
    symbol = atom.GetSymbol()
    print(symbol)'''

'''atom = mol.GetAtomWithIdx(0)
symbol = atom.GetSymbol()
print(symbol)'''

try:
    #img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(300, 300), highlightAtomLists=[[0]])
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(600, 600))
    img.save("draw.png")
except:
    print('error')
    