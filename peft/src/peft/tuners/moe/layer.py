# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, json
import math
import copy
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer

class MoeLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("moe_experts", "moe_router_embedding", "moe_token_classify")
    other_param_names = ("num_experts", "old_num_experts")

    def __init__(self, base_layer: nn.Module) -> None:
        self.base_layer = base_layer
        self.num_experts = {}
        self.old_num_experts = {}
        self.moe_token_classify = nn.ModuleDict({})
        self.moe_router_embedding = nn.ModuleDict({})
        self.moe_experts = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        if hasattr(base_layer, "gate_proj"):
            self.in_features = base_layer.gate_proj.in_features
        elif hasattr(base_layer, "fc1"):
            self.in_features = base_layer.fc1.in_features
        else:
            raise NotImplementedError

    def update_layer(self, base_layer, adapter_name, num_experts, old_num_experts, init_moe_weights, group_nums):
        assert num_experts >= 2  # 最少加一个新专家
        self.num_experts[adapter_name] = num_experts
        self.old_num_experts[adapter_name] = old_num_experts
        if group_nums is not None and group_nums != 0:
            self.moe_token_classify[adapter_name] = nn.Linear(self.in_features, group_nums, bias=False) # tokens分类网络的初始化
        else:
            self.moe_token_classify[adapter_name] = None
        self.moe_router_embedding[adapter_name] = nn.Linear(self.in_features, num_experts, bias=False) # 路由网络的初始化
        self.moe_experts[adapter_name] = nn.ModuleList([copy.deepcopy(base_layer) for _ in range(num_experts - 1)]) # 专家模型的初始化

        if init_moe_weights:
            self.reset_moe_parameters(adapter_name)

        
        if hasattr(base_layer, "gate_proj"):
            weight = base_layer.gate_proj.weight
        elif hasattr(base_layer, "fc1"):
            weight = base_layer.fc1.weight
        else:
            raise NotImplementedError
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_moe_parameters(self, adapter_name):
        if adapter_name in self.moe_router_embedding.keys():
            # initialize A the same way as the default for nn.Linear and B to zero 只初始化moe_router_embedding
            print("$$$$$$$$$$$$$$-reset_moe_parameters")
            nn.init.xavier_normal_(self.moe_router_embedding[adapter_name].weight)
            if self.moe_token_classify[adapter_name] is not None:
                nn.init.xavier_normal_(self.moe_token_classify[adapter_name].weight)


class MLP(nn.Module, MoeLayer):
    # Moe implemented in a mlp layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        num_experts: int = 2,
        old_num_experts: int = None,
        init_moe_weights: bool = True,
        topk: int = None,
        aux_loss_coef: float = None,
        lpr_loss_coef: float = None,
        classify_loss_coef: float = None,
        sequential_add_loss_coef: float = None,
        group_nums: int = None,
        **kwargs,
    ) -> None:
        super().__init__()
        MoeLayer.__init__(self, base_layer)

        self.aux_loss_coef = aux_loss_coef
        self.topk = topk
        self.lpr_loss_coef = lpr_loss_coef
        self.classify_loss_coef = classify_loss_coef
        self.sequential_add_loss_coef = sequential_add_loss_coef
        self._active_adapter = adapter_name

        self.group_nums = group_nums

        self.update_layer(base_layer, adapter_name, num_experts, old_num_experts, init_moe_weights, group_nums)

    def forward(self, x: torch.Tensor, input_ids=None) -> torch.Tensor:
        previous_dtype = x.dtype
        # print("@@@-peft-tuners-moe-layer: line102: self.active_adapter:", self.active_adapter)
        router = self.moe_router_embedding[self.active_adapter[0]]  # b x s x e

        classify_res = None
        routing_info = None
        if self.group_nums is not None and self.group_nums != 0:
            group_classify = self.moe_token_classify[self.active_adapter[0]]
            assert group_classify is not None
            result, router_logits, classify_res, routing_info = self.classify_route_func(x, router, group_classify, self.active_adapter[0])
        else:
            result, router_logits, routing_info = self.topk_route(x, router, self.active_adapter[0])

        result = result.to(previous_dtype)
        return result, router_logits, classify_res, routing_info

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "moe." + rep

    def topk_route(self, hidden_states, router, adapter=None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)  # !!! [batch_size * sequence_length, hidden_dim] !!!

        router_logits = router(hidden_states)


        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        if self.topk > self.num_experts["default"]:
            # adaptive_k setting
            routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts["default"], dim=-1)
        else:
            routing_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)

        if self.topk != 1:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)


        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts[adapter]).permute(2, 1, 0)


        experts = [self.base_layer] + [k for k in self.moe_experts[adapter]]
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts[adapter]):
            expert_layer = experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)   # 将激活expert_idx专家的tokens对应的hidden-states取出，[top_x.size(), hidden_dim]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]  # 按照routing_weights加权


            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # Return routing info for expertise analysis
        routing_info = {
            'selected_experts': selected_experts,  # [batch*seq, topk]
            'routing_weights': routing_weights,    # [batch*seq, topk]
        }
        return final_hidden_states, router_logits, routing_info
    


    def classify_route_func(self, hidden_states, router, group_classify, adapter=None, input_ids=None):

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        softmax = nn.Softmax(dim=-1)

        tmp_classify_logits = group_classify(hidden_states.sum(1))
        tmp_classify_res = softmax(tmp_classify_logits).argmax(dim=-1, keepdim=True)
        classify_res = tmp_classify_res.expand(batch_size, sequence_length).contiguous().view(-1)
        hidden_states = hidden_states.view(-1, hidden_dim)  # !!! [batch_size * sequence_length, hidden_dim] !!!  每一个token的hidden-state

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = router(hidden_states)

        original_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Initialize variables to track routing info across all groups
        all_selected_experts = torch.zeros((batch_size * sequence_length, self.topk), dtype=torch.long, device=hidden_states.device)
        all_routing_weights = torch.zeros((batch_size * sequence_length, self.topk), dtype=hidden_states.dtype, device=hidden_states.device)

        final_hidden_states = torch.zeros(
                    (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
        # for _i_group_id in range(self.group_nums):
        #     cur_group_mask = classify_res.eq(_i_group_id)
        #     if _i_group_id == 0:
        #         # 只路由到 0 专家
        #         cur_final_hidden_states = self.base_layer(hidden_states)
        #         final_hidden_states += cur_final_hidden_states * cur_group_mask.unsqueeze(-1)
        #         # Group 0 always uses expert 0
        #         all_selected_experts[cur_group_mask, :] = 0

        #     elif _i_group_id == 1:
        #         # 路由到 0 专家 + G1 专家；
        #         routing_weights, selected_experts = torch.topk(original_routing_weights[:, :self.old_num_experts[adapter]], self.topk, dim=-1)
        #         if self.topk != 1:
        #             routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        #         routing_weights = routing_weights.to(hidden_states.dtype)
        #         expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.old_num_experts[adapter]).permute(2, 1, 0)
        #         experts = [self.base_layer] + [k for k in self.moe_experts[adapter][:self.old_num_experts[adapter]-1]]
        #         assert len(experts) == self.old_num_experts[adapter]
        #         cur_final_hidden_states = torch.zeros(
        #             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        #             )
        #         for expert_idx in range(self.old_num_experts[adapter]):
        #             expert_layer = experts[expert_idx]
        #             idx, top_x = torch.where(expert_mask[expert_idx])
        #             current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        #             current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        #             cur_final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        #         final_hidden_states += cur_final_hidden_states * cur_group_mask.unsqueeze(-1)
        #         # Store routing info for group 1
        #         all_selected_experts[cur_group_mask] = selected_experts[cur_group_mask]
        #         all_routing_weights[cur_group_mask] = routing_weights[cur_group_mask]
        #     else:
        #         # 路由到所有专家
        #         routing_weights, selected_experts = torch.topk(original_routing_weights, self.topk, dim=-1)

        #         if self.topk != 1:
        #             routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        #         routing_weights = routing_weights.to(hidden_states.dtype)
        #         # One hot encode the selected experts to create an expert mask
        #         # this will be used to easily index which expert is going to be sollicitated
        #         expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts[adapter]).permute(2, 1, 0)
        #         cur_final_hidden_states = torch.zeros(
        #             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        #             )
        #         experts = [self.base_layer] + [k for k in self.moe_experts[adapter]]
        #         # Loop over all available experts in the model and perform the computation on each expert
        #         for expert_idx in range(self.num_experts[adapter]):
        #             expert_layer = experts[expert_idx]
        #             idx, top_x = torch.where(expert_mask[expert_idx])

        #             current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)   # 将激活expert_idx专家的tokens对应的hidden-states取出，[top_x.size(), hidden_dim]
        #             current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]  # 按照routing_weights加权


        #             # However `index_add_` only support torch tensors for indexing so we'll use
        #             # the `top_x` tensor here.
        #             cur_final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        #         final_hidden_states += cur_final_hidden_states * cur_group_mask.unsqueeze(-1)
        #         # Store routing info for group 2+
        #         all_selected_experts[cur_group_mask] = selected_experts[cur_group_mask]
        #         all_routing_weights[cur_group_mask] = routing_weights[cur_group_mask]
        for _i_group_id in range(self.group_nums):
            cur_group_mask = classify_res.eq(_i_group_id)              # (B*S,)
            sel = torch.nonzero(cur_group_mask, as_tuple=False).squeeze(-1)  # (N_g,)
            if sel.numel() == 0:
                continue

            # グループ内のトークンだけ抽出
            hs_g = hidden_states.index_select(0, sel)                  # (N_g, H)
            rw_g = original_routing_weights.index_select(0, sel)       # (N_g, E_total)

            if _i_group_id == 0:
                # base のみ
                y = self.base_layer(hs_g)                               # (N_g, H)
                final_hidden_states.index_copy_(0, sel, y)
                all_selected_experts.index_fill_(0, sel, 0)             # すべて expert=0
                all_routing_weights.index_fill_(0, sel, 1.0 / self.topk if self.topk>0 else 1.0)  # 形だけ埋める

            elif _i_group_id == 1:
                # 旧 expert 群 (0..old_num_experts-1) に限定して top-k
                E = self.old_num_experts[adapter]
                rw_topk, sel_exp = torch.topk(rw_g[:, :E], self.topk, dim=-1)  # (N_g, topk)
                if self.topk != 1:
                    rw_topk /= rw_topk.sum(dim=-1, keepdim=True)
                rw_topk = rw_topk.to(hs_g.dtype)
                expert_mask = F.one_hot(sel_exp, num_classes=E).permute(2, 1, 0)  # (E, topk, N_g)
                experts = [self.base_layer] + [k for k in self.moe_experts[adapter][:E-1]]

                cur_out = hs_g.new_zeros((hs_g.size(0), hidden_dim))
                for e_idx in range(E):
                    idx, top_local = torch.where(expert_mask[e_idx])    # (topk_pos, token_local)
                    if top_local.numel() == 0:
                        continue
                    x_e = hs_g.index_select(0, top_local)               # (n_e, H)
                    y_e = experts[e_idx](x_e) * rw_topk[top_local, idx].unsqueeze(-1)
                    cur_out.index_add_(0, top_local, y_e)
                final_hidden_states.index_copy_(0, sel, cur_out)
                all_selected_experts.index_copy_(0, sel, sel_exp)
                all_routing_weights.index_copy_(0, sel, rw_topk)

            else:
                # 全 expert を対象
                E = self.num_experts[adapter]
                rw_topk, sel_exp = torch.topk(rw_g, self.topk, dim=-1)
                if self.topk != 1:
                    rw_topk /= rw_topk.sum(dim=-1, keepdim=True)
                rw_topk = rw_topk.to(hs_g.dtype)
                expert_mask = F.one_hot(sel_exp, num_classes=E).permute(2, 1, 0)  # (E, topk, N_g)
                experts = [self.base_layer] + [k for k in self.moe_experts[adapter]]

                cur_out = hs_g.new_zeros((hs_g.size(0), hidden_dim))
                for e_idx in range(E):
                    idx, top_local = torch.where(expert_mask[e_idx])
                    if top_local.numel() == 0:
                        continue
                    x_e = hs_g.index_select(0, top_local)
                    y_e = experts[e_idx](x_e) * rw_topk[top_local, idx].unsqueeze(-1)
                    cur_out.index_add_(0, top_local, y_e)
                final_hidden_states.index_copy_(0, sel, cur_out)
                all_selected_experts.index_copy_(0, sel, sel_exp)
                all_routing_weights.index_copy_(0, sel, rw_topk)


        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # Return routing info for expertise analysis
        routing_info = {
            'selected_experts': all_selected_experts,  # [batch*seq, topk]
            'routing_weights': all_routing_weights,    # [batch*seq, topk]
            'classify_res': classify_res,              # [batch*seq]
        }
        return final_hidden_states, router_logits, tmp_classify_logits, routing_info