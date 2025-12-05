import os
import time
import argparse

import torch
import torch.nn.functional as F
from rdkit import Chem
import cv2

from molsight.dataset import get_transforms
from molsight.model import MolsightModel, get_edge_prediction
from molsight.tokenizer import CharTokenizer, SOS_ID, EOS_ID, PAD_ID
from molsight.chemistry import _postprocess_smiles

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--encoder', type=str, default='efficientvit', choices=['efficientvit', 'vitdet'])
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument("--dec_n_layer", help="No. of layers in transformer decoder", type=int, default=6)
    parser.add_argument("--dec_n_head", help="Decoder no. of attention heads", type=int, default=8)
    parser.add_argument("--use_qknorm", type=bool, default=True, help="Use QKNorm in the decoder")
    parser.add_argument("--use_swiglu", type=bool, default=True, help="Use SwiGLU in the decoder")
    parser.add_argument("--use_rmsnorm", type=bool, default=True, help="Use RMSNorm in the decoder")
    parser.add_argument('--lora', action='store_true', help='Use LoRA for the decoder.')
    parser.add_argument('--regression', type=bool, default=False, help='Use regression for coordinate prediction.')
    # Data
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--formats', type=str, default='char,edges')
    parser.add_argument('--vocab_file', type=str, default='vocab/vocab_chars.json')
    parser.add_argument('--resume', type=bool, default=True, help='Resume training from the last checkpoint.')
    parser.add_argument('--max_len', type=int, default=320)
    # Inference
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--save_attns', action='store_true')
    parser.add_argument('--molblock', action='store_true')
    parser.add_argument('--compute_confidence', action='store_true')
    parser.add_argument('--keep_main_molecule', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    args.formats = args.formats.split(',')

    tokenizer = CharTokenizer(args.vocab_file)

    model = MolsightModel(args, tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # load checkpoint
    ckpt_path = 'pubchem_uspto_smiles_edges_30.pth'
    if not os.path.exists(ckpt_path):
        import urllib.request
        weight_url = 'https://huggingface.co/Robert-zwr/MolSight/blob/main/pubchem_uspto_smiles_edges_30.pth?download=true'
        # if you cannot access huggingface, use the mirror link below
        #weight_url = 'https://hf-mirror.com/Robert-zwr/MolSight/resolve/main/pubchem_uspto_smiles_edges_30.pth?download=true'
        print(f'Downloading model weights from {weight_url} ...')
        urllib.request.urlretrieve(weight_url, ckpt_path)
        print('Download completed.')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_checkpoint = checkpoint["model"]

    # For DDP, we need to load the model state dict from the module
    model_checkpoint = {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in model_checkpoint.items()
    }
    missing_keys, unexpected_keys = model.load_state_dict(model_checkpoint, strict=False)
    print(f'Missing keys: {missing_keys}')
    print(f'Unexpected keys: {unexpected_keys}')
    
    # switch to evaluation mode
    if hasattr(model, 'module'):
        model = model.module
    model.eval()

    image_path = 'assets/example.png'  # path to the input image
    assert os.path.exists(image_path), f'Image file {image_path} does not exist.'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
    transform = get_transforms(args, augment=False, rotate=False)
    augmented = transform(image=image)
    image = augmented['image'].unsqueeze(0)  # [1, C, H, W]
    image = image.to(device)
    n_img = 1
    
    with torch.no_grad():
        # install kv_cache hooks
        kv_cache, hooks = model.install_kv_cache_hooks()

        start = time.time()
        batch_preds, inter = model.generate(image=image, kv_cache=kv_cache)
        end = time.time()
        print(f'Inference time: {end - start:.2f} seconds')

        xi = inter['image_features']    # [n_img, H*W, hidden_dim]
        hidden_states = inter['hidden_states']  # [n_img, max_len, hidden_dim]
        
        # remove hooks
        for hook in hooks:
            hook.remove()

        edge_preds = edge_scores = loc_preds = None
        if 'edges' in args.formats:
            atom_indices = [torch.LongTensor(i) for i in batch_preds["indices"]]
            atom_indices = torch.nn.utils.rnn.pad_sequence(atom_indices, batch_first=True, padding_value=PAD_ID).to(hidden_states.device)   # [n_img, max_len]
            atom_indices = atom_indices + model.sample_begin  # convert to the indices in the hidden states
            
            edge_logits = model.decoder.edge_predictor(hidden_states, atom_indices)    # [b, l, l, 7]
            edge_probs = F.softmax(edge_logits, dim=-1)
            valid_lengths = [len(ind) for ind in batch_preds["indices"]]
            edge_preds, edge_scores = get_edge_prediction(edge_probs, valid_lengths)
        if 'coords' in args.formats:
            seq_len = hidden_states.size(1)
            position_embeddings = (model.decoder.embed_cos[:, :seq_len].to(xi.dtype), model.decoder.embed_sin[:, :seq_len].to(xi.dtype)) if args.use_qknorm else None
            for block in model.decoder.loc_predictor.loc_blocks:
                hidden_states = block(hidden_states, xi, position_embeddings=position_embeddings)
            loc_coords, x_hms, y_hms, sigmas = model.decoder.loc_predictor(hidden_states, atom_indices)
            #loc_coords = model.decoder.loc_predictor(hidden_states, atom_indices)[0]  # [b, l, 2]

            loc_preds = [loc_coords[i, :int(valid_lengths[i])].tolist() for i in range(n_img)]

        batch_preds['edges'] = edge_preds
        batch_preds['edge_scores'] = edge_scores
        batch_preds['coords'] = loc_preds

        print('SMILES: ', batch_preds['smiles'][0])
        print('atoms: ', batch_preds['atoms'][0])
        print('indices :', batch_preds['indices'][0])

        SMILES, _, success = _postprocess_smiles(batch_preds['smiles'][0])
        SMILES = Chem.CanonSmiles(SMILES)
        print('Postprocessed SMILES: ', SMILES)


if __name__ == "__main__":
    main()
