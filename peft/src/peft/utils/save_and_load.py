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
import os
import warnings
from typing import Optional

import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import load_file as safe_load_file

from .other import EMBEDDING_LAYER_NAMES, SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, infer_device
from .peft_types import PeftType


def has_valid_embedding_base_layer(layer):
    """Check if the layer has an embedding base layer"""
    return hasattr(layer, "base_layer") and isinstance(layer.base_layer, (torch.nn.Linear, torch.nn.Embedding))


def get_embedding_layer_name(model, layer, is_embedding_in_target_modules):
    """Get the name of the embedding module for a given layer."""
    for name, module in model.named_modules():
        if (not is_embedding_in_target_modules and module == layer) or module == getattr(layer, "base_layer", None):
            return name
    return None


def get_peft_model_state_dict(
    model, state_dict=None, adapter_name="default", unwrap_compiled=False, save_embedding_layers="auto"
):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be returned.
        unwrap_compiled (`bool`, *optional*, defaults to `False`):
            Whether to unwrap the model if torch.compile was used.
        save_embedding_layers (`Union[bool, str]`, , *optional*, defaults to `auto`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common embedding
            layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available. Based on it
            sets the boolean flag. This only works for 🤗 transformers models.
    """
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type is PeftType.MOE:
        if config.save_all_params:
            return state_dict
        else:
            to_return = {k: state_dict[k] for k in state_dict if "moe_" in k}
            to_return = {k: v for k, v in to_return.items() if (("moe_" in k and adapter_name in k))}
    elif config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

    elif config.peft_type == PeftType.LOHA:
        to_return = {k: state_dict[k] for k in state_dict if "hada_" in k}

    elif config.peft_type == PeftType.LOKR:
        to_return = {k: state_dict[k] for k in state_dict if "lokr_" in k}

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif config.is_prompt_learning:
        to_return = {}
        if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            to_return["prefix_task_cols"] = model.prompt_encoder[adapter_name].prefix_task_cols
            to_return["prefix_task_rows"] = model.prompt_encoder[adapter_name].prefix_task_rows
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            if config.inference_mode:
                prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
            else:
                prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
    elif config.peft_type == PeftType.IA3:
        to_return = {k: state_dict[k] for k in state_dict if "ia3_" in k}
    elif config.peft_type == PeftType.OFT:
        to_return = {k: state_dict[k] for k in state_dict if "oft_" in k}
    elif config.peft_type == PeftType.POLY:
        to_return = {k: state_dict[k] for k in state_dict if "poly_" in k}
    else:
        raise NotImplementedError
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    # check the common embedding layers in `target_modules` to reset `save_embedding_layers` if necessary
    is_embedding_in_target_modules = False
    if (
        save_embedding_layers == "auto"
        and hasattr(config, "target_modules")
        and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES)
    ):
        warnings.warn("Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == "auto":
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        model_id = getattr(config, "base_model_name_or_path", None)

        # For some models e.g. diffusers the text config file is stored in a subfolder
        # we need to make sure we can download that config.
        has_remote_config = False

        if model_id is not None:
            try:
                has_remote_config = file_exists(model_id, "config.json")
            except (HFValidationError, EntryNotFoundError):
                warnings.warn(
                    f"Could not find a config file in {model_id} - will assume that the vocabulary was not modified."
                )
                has_remote_config = False

        # check if the vocab size of the base model is different from the vocab size of the finetuned model
        if (
            vocab_size
            and model_id
            and has_remote_config
            and (vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size)
        ):
            warnings.warn(
                "Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning."
            )
            save_embedding_layers = True
        else:
            save_embedding_layers = False

    if save_embedding_layers and hasattr(model, "get_input_embeddings"):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                # support from version >= 0.6.2
                embedding_module_name = get_embedding_layer_name(model, layer, is_embedding_in_target_modules)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn("Could not identify embedding layer(s) because the model is not a 🤗 transformers model.")

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    sum_experts_new = 0
    num_experts_config = getattr(config, 'num_experts', 1)

    for layer_idx, _tmp_layer in enumerate(model.base_model.model.model.layers):
        # Determine expected expert count for this layer
        if isinstance(num_experts_config, list):
            expected_experts = num_experts_config[layer_idx] if layer_idx < len(num_experts_config) else 1
        else:
            expected_experts = num_experts_config

        if expected_experts == 1:
            # Single expert = original MLP layer (intentionally non-MoE)
            # sum_experts_new += 1
            # Do not count non-MoE layers
            continue
        elif hasattr(_tmp_layer.mlp, 'num_experts'):
            # MoE layer with multiple experts - get actual count
            sum_experts_new += int(_tmp_layer.mlp.num_experts[adapter_name])
        else:
            # Layer not yet converted to MoE but should be - use config value
            sum_experts_new += expected_experts

    print("&&&sum_experts_new: ", sum_experts_new)
    sum_experts_old = 0
    for key, value in peft_model_state_dict.items():
            if "router_embedding" in key:
                sum_experts_old += value.size(0)
    # 判断新模型的专家总数和旧模型的专家总数是否一致即可，如果一致则使用之前的 load 方式；如果不一致，则路由网络需要重新路由
    state_dict = {}
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict
    print("&&&sum_experts_old: ", sum_experts_old)
    if config.peft_type in (
        PeftType.LORA,
        PeftType.MOE,
        PeftType.LOHA,
        PeftType.LOKR,
        PeftType.ADALORA,
        PeftType.IA3,
        PeftType.OFT,
        PeftType.POLY,
    ):
        
        peft_model_state_dict = {}
        peft_model_state_dict_no_router = {}
        peft_model_state_dict_no_classify = {}
        peft_model_state_dict_router_old = {}
        parameter_prefix = {
            PeftType.IA3: "ia3_",
            PeftType.LORA: "lora_",
            PeftType.ADALORA: "lora_",
            PeftType.LOHA: "hada_",
            PeftType.LOKR: "lokr_",
            PeftType.OFT: "oft_",
            PeftType.POLY: "poly_",
            PeftType.MOE: "moe_",
        }[config.peft_type]
        for k, v in state_dict.items():
            if parameter_prefix in k:
                print("@@@-peft-utils-set_peft_model_state_dict (lines231)", k)
                suffix = k.split(parameter_prefix)[1]
                # print("@@@@@-suffix:", suffix)   # _experts.0.down_proj.weight
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                    # print("!!!!!!!!")
                else:
                    k = f"{k}.{adapter_name}"
                    # print("????????")
                peft_model_state_dict[k] = v
                if "classify" not in k:
                    peft_model_state_dict_no_classify[k] = v
                
                if "router" not in k:
                    peft_model_state_dict_no_router[k] = v
                else:
                    peft_model_state_dict_router_old[k] = v
            else:
                peft_model_state_dict[k] = v
                if "classify" not in k:
                    peft_model_state_dict_no_classify[k] = v
                if "router" not in k:
                    peft_model_state_dict_no_router[k] = v
                else:
                    peft_model_state_dict_router_old[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    if sum_experts_old == sum_experts_new:
        print("***** Router embedding & experts are loaded! *****")
        try:
            load_result = model.load_state_dict(peft_model_state_dict, strict=False)
            print("***** Router embedding & experts & classify are loaded! *****")
        except:
            load_result = model.load_state_dict(peft_model_state_dict_no_classify, strict=False)
            print("***** Router embedding & experts are loaded! (no classify) *****")
    else:
        print("!!!!! Router embedding is not loaded (Since this is the sequential-adding setting.) only in stage-1 !!!!!")
        # print(peft_model_state_dict_no_router.keys())  # base_model.model.model.layers.4.mlp.moe_experts.default.1.down_proj.weight
        load_result = model.load_state_dict(peft_model_state_dict_no_router, strict=False)
        
        for n, p in model.named_parameters():
                # print(n)  # base_model.model.model.layers.23.mlp.moe_experts.default.3.gate_proj.weight
                if "router" in n or "classify" in n:
                    #这里还需要用 old 的 router embedding 初始化 new 的 router embedding 吗？ 这里可以再考虑一下；感觉意义似乎不大，stage2 本身还要重新训练路由网络
                    # p.data[:4, :] = peft_model_state_dict_router_old[n]
                    # print(p.shape, p.data[:4, :].shape, peft_model_state_dict_router_old[n].size())

                    p.requires_grad = True
                elif "experts" in n and n not in list(peft_model_state_dict_no_router.keys()):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        
    if config.is_prompt_learning:
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )

    if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
        model.prompt_encoder[adapter_name].load_state_dict(peft_model_state_dict, strict=False)

    print("@@@-peft-utils-set_peft_model_state_dict (line325):", type(load_result))
    return load_result


def load_peft_weights(model_id: str, device: Optional[str] = None, **hf_hub_download_kwargs) -> dict:
    r"""
    A helper method to load the PEFT weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading from the HuggingFace Hub.
    """
    path = (
        os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
        if hf_hub_download_kwargs.get("subfolder", None) is not None
        else model_id
    )

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        token = hf_hub_download_kwargs.get("token", None)
        if token is None:
            token = hf_hub_download_kwargs.get("use_auth_token", None)

        has_remote_safetensors_file = file_exists(
            repo_id=model_id,
            filename=SAFETENSORS_WEIGHTS_NAME,
            revision=hf_hub_download_kwargs.get("revision", None),
            repo_type=hf_hub_download_kwargs.get("repo_type", None),
            token=token,
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                )

    if use_safetensors:
        if hasattr(torch.backends, "mps") and (device == torch.device("mps")):
            adapters_weights = safe_load_file(filename, device="cpu")
        else:
            adapters_weights = safe_load_file(filename, device="cpu")
    else:
        adapters_weights = torch.load(filename, map_location=torch.device(device))

    print("@@@-peft-utils-load_peft_weights form (line325):", filename)
    return adapters_weights
