from typing import Dict
import torch, json
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        balance_loss = outputs.balance_loss
        lpr_loss = outputs.lpr_loss
        lang_tag_loss = outputs.lang_tag_loss
        share_balance_loss = outputs.share_balance_loss
        sequential_add_loss = outputs.sequential_add_loss
        classify_loss = outputs.classify_loss

        if balance_loss or share_balance_loss != None:
            nlayers = len(outputs.router_logits)
            prefix = "load_balance_"
            with torch.no_grad():
                mask = inputs['attention_mask'].reshape(-1) # n
                router_logits = outputs.router_logits
                router_logits_layers_dict = {}
                router_logits_layers_all = []  # Store all layers with their indices

                for layer_idx, ilayer_router_logits in enumerate(router_logits):
                    cur_experts_num = ilayer_router_logits.shape[-1]
                    router_logits_layers_all.append((layer_idx, cur_experts_num, ilayer_router_logits))

                    if cur_experts_num not in router_logits_layers_dict:
                        router_logits_layers_dict[cur_experts_num] = (ilayer_router_logits,)
                    else:
                        router_logits_layers_dict[cur_experts_num] += (ilayer_router_logits,)

                probs = {}
                probs_layers = {}
                probs_all_layers = {}  # Store scores for each individual layer

                for i_key in router_logits_layers_dict.keys():
                    cur_router_logits = torch.stack(router_logits_layers_dict[i_key])
                    cur_mask = mask.unsqueeze(0).expand(cur_router_logits.shape[0], mask.size(0)).bool()
                    cur_probs = torch.nn.functional.softmax(cur_router_logits, dim=-1)
                    probs[i_key] = torch.mean(cur_probs[cur_mask], dim=0).detach().cpu().tolist()
                    probs_layers[i_key] = torch.mean(cur_probs[cur_mask].reshape(cur_probs.size(0), -1, cur_probs.size(-1)), dim=1).detach().cpu().tolist()

                # Calculate scores for all individual layers
                for layer_idx, cur_experts_num, ilayer_router_logits in router_logits_layers_all:
                    if cur_experts_num > 1:  # Only process MoE layers
                        cur_mask = mask.bool()
                        cur_probs = torch.nn.functional.softmax(ilayer_router_logits, dim=-1)
                        layer_probs = torch.mean(cur_probs[cur_mask], dim=0).detach().cpu().tolist()
                        probs_all_layers[layer_idx] = (cur_experts_num, layer_probs)

            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = balance_loss.item() if balance_loss is not None else share_balance_loss.item()
                for i_key in probs.keys():
                    # Log as individual expert scores for WandB compatibility
                    expert_scores = [round(k, 2) for k in probs[i_key]]
                    for expert_idx, score in enumerate(expert_scores):
                        logs[f"scores_per_expert_{i_key}/expert_{expert_idx}"] = score
                    # Keep original string format for console logging
                    logs[f"scores_per_expert_{i_key}_str"] = " ".join([str(score) for score in expert_scores])

                for i_key in probs_layers.keys():
                    for j_layer, _out_probs in enumerate(probs_layers[i_key]):
                        # Log as individual expert scores for WandB compatibility
                        layer_expert_scores = [round(k, 2) for k in _out_probs]
                        for expert_idx, score in enumerate(layer_expert_scores):
                            logs[f"scores_per_expert_{i_key}_layer_{j_layer}/expert_{expert_idx}"] = score
                        # Keep original string format for console logging
                        logs[f"scores_per_expert_{i_key}_cnt-layer_{j_layer}_str"] = " ".join([str(score) for score in layer_expert_scores])

                # Log all individual layer scores with actual layer indices (only MoE layers with >1 expert)
                for layer_idx, (cur_experts_num, layer_probs) in probs_all_layers.items():
                    layer_expert_scores = [round(k, 2) for k in layer_probs]
                    for expert_idx, score in enumerate(layer_expert_scores):
                        logs[f"scores_per_expert_{cur_experts_num}_layer_{layer_idx}/expert_{expert_idx}"] = score
                    # Keep original string format for console logging
                    logs[f"scores_per_expert_{cur_experts_num}_layer_{layer_idx}_str"] = " ".join([str(score) for score in layer_expert_scores])

                self.log(logs)

        if lpr_loss != None:
            prefix = "lpr_"
            lang_mask = inputs['langs']
            with torch.no_grad():
                router_logits = outputs.router_logits

                router_logits_layers_dict = {}
                for ilayer_router_logits in router_logits:
                    cur_experts_num = ilayer_router_logits.shape[-1]
                    if cur_experts_num not in router_logits_layers_dict:
                        router_logits_layers_dict[cur_experts_num] = (ilayer_router_logits,)
                    else:
                        router_logits_layers_dict[cur_experts_num] += (ilayer_router_logits,)

                probs = {}
                for i_key in router_logits_layers_dict.keys():
                    cur_router_logits = torch.stack(router_logits_layers_dict[i_key])
                    cur_router_probs = torch.nn.functional.softmax(cur_router_logits, dim=-1)
                    cur_mask = lang_mask.reshape(-1).bool().expand(cur_router_probs.size()[:2])
                    cur_probs = cur_router_probs[cur_mask].to(torch.float).reshape(cur_mask.size(0), -1, cur_router_probs.size(-1))
                    cur_probs = cur_probs[:, :, 0]
                    probs[i_key] = torch.mean(cur_probs, dim=-1).detach().cpu()


            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = lpr_loss.item()
                for i_key in probs.keys():
                    logs[f"old_lang_expert0_score_expert_{i_key}"] = " ".join([str(round(k, 2)) for k in probs[i_key].tolist()])
                self.log(logs)

        if sequential_add_loss != None:
            prefix = "sequential_adding_loss_"
            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                if sequential_add_loss != []:
                    logs[f"{prefix}_loss"] = sequential_add_loss.item()
                    self.log(logs)
        
        if classify_loss != None:
            prefix = "classify_loss_"
            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = classify_loss.item()
                self.log(logs)
        
                 
        if self.main_loss_logged: self.main_loss_logged = False

        return loss
    

