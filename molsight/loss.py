import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .tokenizer import EOS_ID, PAD_ID, MASK, MASK_ID
from .decoding import GreedyDecoder
from .model import enable_lora, disable_lora


class SmoothedKLDLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=None):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(SmoothedKLDLoss, self).__init__()

        if ignore_index is not None:
            smoothing_value = label_smoothing / (tgt_vocab_size - 1)
        else:
            smoothing_value = label_smoothing / tgt_vocab_size
        uniform = torch.full((tgt_vocab_size,), smoothing_value)
        if ignore_index is not None:
            uniform[ignore_index] = 0
        self.register_buffer('uniform', uniform.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # assuming output is raw logits
        # convert to log_probs
        pred_log_prob = F.log_softmax(output, dim=-1)

        target_prob = self.uniform.repeat(target.size(0), 1)
        target_prob.scatter_(1, target.unsqueeze(1), self.confidence)

        if self.ignore_index is not None:
            mask = target.eq(self.ignore_index)
            target_prob.masked_fill_(mask.unsqueeze(1), 0)
            n_valid = max(target.size(0) - mask.sum().item(), 1)
            loss = F.kl_div(pred_log_prob, target_prob, reduction='sum') / n_valid
        else:
            loss = F.kl_div(pred_log_prob, target_prob, reduction='batchmean')

        return loss


class SequenceLoss(nn.Module):

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, ignore_indices=[]):
        super(SequenceLoss, self).__init__()
        if ignore_indices:
            ignore_index = ignore_indices[0]
        self.ignore_index = ignore_index
        self.ignore_indices = ignore_indices
        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        else:
            self.criterion = SmoothedKLDLoss(label_smoothing, vocab_size, ignore_index)

    def forward(self, output, target):
        """
        :param output: [batch, len, vocab]
        :param target: [batch, len]
        :return:
        """
        batch_size, max_len, vocab_size = output.size()
        output = output[:, :-1].reshape(-1, vocab_size)
        target = target[:, 1:].reshape(-1)
        for idx in self.ignore_indices:
            if idx != self.ignore_index:
                target.masked_fill((target == idx), self.ignore_index)
        loss = self.criterion(output, target)
        return loss


class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()
        weight = torch.ones(7, dtype=torch.bfloat16)
        weight[0] = 0.1
        self.criterion = nn.CrossEntropyLoss(weight, ignore_index=-100)

    def forward(self, preds, targets):
        preds = preds.view(-1, preds.size(-1))  # [batch * num_edges, num_classes]
        targets = targets.view(-1)  # [batch * num_edges]
        edge_loss = self.criterion(preds, targets)
        edge_loss[torch.isnan(edge_loss)] = 0.0
        mask = targets.ge(0)
        edge_loss = (edge_loss * mask).sum() / (mask.sum() + 1e-5)
        return edge_loss
    

class CoordMLELoss(nn.Module):

    def __init__(self, args):
        super(CoordMLELoss, self).__init__()
        self.register_buffer('x_bins', torch.linspace(0, 1, args.n_coord_bins), persistent=False)
        self.register_buffer('y_bins', torch.linspace(0, 1, args.n_coord_bins), persistent=False)

    def forward(self, x_hms, y_hms, sigmas, targets):
        '''
        x_hms: [batch, num_atoms, n_bins]
        y_hms: [batch, num_atoms, n_bins]
        sigmas: [batch, num_atoms, 1]
        targets: [batch, num_atoms, 2]
        '''
        dist_x = torch.abs(targets.narrow(2, 0, 1) - self.x_bins)  # [batch, num_atoms, n_bins]
        dist_y = torch.abs(targets.narrow(2, 1, 1) - self.y_bins)

        hm_x = torch.exp(-dist_x / sigmas) / (2 * sigmas)   # Laplace distribution
        hm_y = torch.exp(-dist_y / sigmas) / (2 * sigmas)

        denom_x = hm_x.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        denom_y = hm_y.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        hm_x = hm_x / denom_x
        hm_y = hm_y / denom_y

        prob_x = (x_hms * hm_x).sum(dim=-1)  # [batch, num_atoms]
        prob_y = (y_hms * hm_y).sum(dim=-1)

        loss = -torch.log((prob_x * prob_y) + 1e-4) / 2
        loss[torch.isnan(loss)] = 0.0
        mask = targets[:, :, 0].ge(0) # [batch, num_atoms]
        loss = (loss * mask).sum() / (mask.sum() + 1e-5)

        return loss
    

class CoordLoss(nn.Module):

    def __init__(self, args, aux_loss=False):
        super(CoordLoss, self).__init__()
        self.weight = {
                'mle': 1.0,
                'kld': 1.0
            }
        self.loss_mle = CoordMLELoss(args)
        self.aux_loss = aux_loss
        if aux_loss:
            self.loss_kld = SmoothedKLDLoss(args.label_smoothing, args.n_coord_bins)
        self.regression = args.regression

    def forward(self, pred, targets):
        '''
        pred_coords: Tuple of (loc_pred, x_hms, y_hms, sigmas)
        targets: [batch, num_atoms, 2]
        '''        
        loc_pred, x_hms, y_hms, sigmas = pred

        if self.regression:
            # loc_pred: [batch, num_atoms, 2]
            # targets: [batch, num_atoms, 2]
            loss = F.l1_loss(loc_pred, targets, reduction='none')   # [batch, num_atoms, 2]
            loss[torch.isnan(loss)] = 0.0
            mask = targets.ge(0)
            return (loss * mask).sum() / (mask.sum() + 1e-5)

        loss_mle = self.loss_mle(x_hms, y_hms, sigmas, targets)
        
        if self.aux_loss:
            x_hms = x_hms.view(-1, x_hms.size(-1))  # [batch * num_atoms, n_bins]
            y_hms = y_hms.view(-1, y_hms.size(-1))  # [batch * num_atoms, n_bins]
            targets = targets.view(-1, 2)  # [batch * num_atoms, 2]
            loss_kld = self.loss_kld(x_hms, targets[:, 0]) + self.loss_kld(y_hms, targets[:, 1])           
            return self.weight['mle'] * loss_mle + self.weight['kld'] * loss_kld

        return self.weight['mle'] * loss_mle


class Criterion(nn.Module):

    def __init__(self, args, tokenizer):
        super(Criterion, self).__init__()
        criterion = {}
        if 'char' in args.formats:
            if MASK in tokenizer.stoi:
                ignore_indices = [PAD_ID, MASK_ID]
            else:
                ignore_indices = []
            criterion['sequence'] = SequenceLoss(args.label_smoothing, len(tokenizer),
                                                ignore_index=PAD_ID, ignore_indices=ignore_indices)
        if 'edges' in args.formats:
            criterion['edges'] = EdgeLoss()
        if 'coords' in args.formats:
            criterion['coords'] = CoordLoss(args)
        self.criterion = nn.ModuleDict(criterion)

    def forward(self, logits, edge_pred, loc_pred, label, edges, coords, **kwargs):
        losses = {}
        with torch.cuda.amp.autocast(enabled=False):
            for format in self.criterion.keys():
                if format == 'sequence':
                    losses[format] = self.criterion[format](logits, label)
                elif format == 'edges':
                    losses[format] = self.criterion[format](edge_pred, edges)
                elif format == 'coords':
                    losses[format] = self.criterion[format](loc_pred, coords)
                else:
                    raise NotImplementedError
        
        return losses


class SCSTLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.baseline_decode_strategy = GreedyDecoder(temperature=0, eot=EOS_ID)
        self.stochastic_decode_strategy = GreedyDecoder(temperature=1.5, top_k=30, eot=EOS_ID)

    def forward(self, model, image, reward_fn):
        model.eval()  # sampling 时不使用 dropout
        with torch.inference_mode():
            image_features = model.module.get_image_features(image)
            # baseline：贪婪生成，作为 baseline（用于 SCST）
            kv_cache, hooks = model.module.install_kv_cache_hooks()
            greedy_preds, _ = model.module.generate(
                image=image, 
                image_features=image_features,
                kv_cache=kv_cache, 
                decode_strategy=self.baseline_decode_strategy
            )
            for hook in hooks:
                hook.remove()
                
            greedy_smiles = greedy_preds['smiles']
            greedy_rewards = reward_fn(greedy_smiles)

            # 采样生成（stochastic）
            kv_cache, hooks = model.module.install_kv_cache_hooks()
            sampled_preds, inter = model.module.generate(
                image=image, 
                image_features=image_features,
                kv_cache=kv_cache, 
                decode_strategy=self.stochastic_decode_strategy,
                n_samples=self.args.n_samples
            )
            for hook in hooks:
                hook.remove()

            sampled_ids = inter['raw_tokens']
            
            sampled_smiles = sampled_preds['smiles']
            sampled_rewards = reward_fn(sampled_smiles)

        # TODO: deal with running stats changing in BatchNorm
        model.train()

        # 获取 log probs
        image_features = image_features.repeat_interleave(self.args.n_samples, dim=0)
        _, logits, _, _ = model.module.decoder(sampled_ids[:, :-1], image_features.flatten(2).permute(0, 2, 1))
        #outputs = model(image=image, label=sampled_ids[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)    # (B*G, T, V)
        
        target_ids = sampled_ids[:, 1:]  # (n_samples, T)

        valid_token_mask = torch.ones_like(target_ids, dtype=torch.bool)
        for i in range(target_ids.size(0)):
            eos_pos = (target_ids[i] == EOS_ID).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                valid_token_mask[i, eos_pos[0] + 1:] = False  # mask 掉 EOS 之后的

        sampled_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)    # (B, T)

        masked_log_probs = sampled_log_probs * valid_token_mask.float()  # (B, T)

        seq_lengths = valid_token_mask.sum(dim=1)  # (B,)
        norm_log_probs = masked_log_probs.sum(dim=1) / seq_lengths.clamp(min=1)

        # 计算 advantage
        advantage = sampled_rewards - greedy_rewards  # shape: (B,)
        rl_loss = - (norm_log_probs * advantage).mean()

        return rl_loss, {
            'avg_reward': sampled_rewards.mean().item(),
            'avg_baseline': greedy_rewards.mean().item(),
        }

class GRPOLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.beta = 0.04
        self.args = args
        #self.baseline_decode_strategy = GreedyDecoder(temperature=0, eot=EOS_ID)
        self.stochastic_decode_strategy = GreedyDecoder(temperature=1.0, top_k=10, eot=EOS_ID)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, image_features):
        image_features = image_features.repeat_interleave(self.args.n_samples, dim=0)
        logits = model(image=None, label=input_ids, image_features=image_features)["logits"]
        #logits = model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def forward(self, model, image, reward_fn):
        enable_lora(model.module)  # enable LoRA for sampling

        with torch.no_grad():
            image_features = model.module.get_image_features(image)
            # Generate completions
            kv_cache, hooks = model.module.install_kv_cache_hooks()
            sampled_preds, inter = model.module.generate(
                image=image, 
                image_features=image_features,
                kv_cache=kv_cache, 
                decode_strategy=self.stochastic_decode_strategy,
                n_samples=self.args.n_samples
            )
            for hook in hooks:
                hook.remove()

        sampled_ids = inter['raw_tokens']
        target_ids = sampled_ids[:, 1:]  # (B*G, T-1)
        valid_token_mask = torch.ones_like(target_ids, dtype=torch.bool)
        for i in range(target_ids.size(0)):
            eos_pos = (target_ids[i] == EOS_ID).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                valid_token_mask[i, eos_pos[0] + 1:] = False  # Mask everything after the first EOS token

        # get log probs
        per_token_logps = self._get_per_token_logps(model, sampled_ids, image_features)  # (B*G, T-1)

        with torch.inference_mode():
            disable_lora(model.module)
            ref_per_token_logps = self._get_per_token_logps(
                model.module,
                sampled_ids.detach(),
                image_features.detach()
            )
        
        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Compute the rewards
        rewards = reward_fn(sampled_preds['smiles'])    # (B*G,)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.args.n_samples).mean(dim=1)    # (B,)
        std_grouped_rewards = rewards.view(-1, self.args.n_samples).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.args.n_samples, dim=0)   # (B*G,)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.args.n_samples, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * valid_token_mask).sum(dim=1) / valid_token_mask.sum(dim=1)).mean()

        mean_kl = ((per_token_kl * valid_token_mask).sum(dim=1) / valid_token_mask.sum(dim=1)).mean()
        return loss, {
            'avg_rewards': rewards.mean().item(),
            'rewards_std': std_grouped_rewards.mean().item(),
            'kl': mean_kl.mean().item(),
        }
