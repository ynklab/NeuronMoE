import json, os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, torch
from tqdm import tqdm
from types import MethodType
from torch.nn import functional as F
from peft import PeftModel, PeftConfig
import jsonlines


def main(model, tokenizer, langs, data_path_dict, device="cuda:0"):
    num_layers = model.config.num_hidden_layers

    tokens_limit = 150000
    tokens_select = 100000

    all_langs_hidden_states = {}
    for lang in langs.split("_"):
        data_path = data_path_dict[lang]
        data_select = []
        all_tokens = 0
        with jsonlines.open(data_path) as f:
            for line in f:
                if all_tokens > tokens_limit: break
                tmp_tok_res = tokenizer(line["text"])["input_ids"]
                if len(tmp_tok_res) > 32000: continue  # Skip very long sequences to avoid OOM
                all_tokens += len(tmp_tok_res)
                data_select.append(line["text"])
        print(f"lang: {lang}, data_num: {len(data_select)}, all_tokens: {all_tokens}")

        global hidden_states_lang
        hidden_states_lang = [[] for i in range(num_layers)]

        def factory(idx):
            def new_forward(self, x: torch.Tensor, input_ids=None) -> torch.Tensor:
                hidden_states_lang[idx].append(x.reshape(-1, x.size(-1)).cpu())
                previous_dtype = x.dtype
                router = self.moe_router_embedding[self.active_adapter[0]]

                classify_res = None
                if self.group_nums is not None and self.group_nums != 0:
                    group_classify = self.moe_token_classify[self.active_adapter[0]]
                    assert group_classify is not None
                    result, router_logits, classify_res = self.classify_route_func(x, router, group_classify, self.active_adapter[0])
                else:
                    result, router_logits = self.topk_route(x, router, self.active_adapter[0])

                result = result.to(previous_dtype)
                return result, router_logits, classify_res

            return new_forward

        for i in range(num_layers):
            obj = model.base_model.model.model.layers[i].mlp
            obj.forward = MethodType(factory(i), obj)

        for i in range(len(data_select)):
            prompt = data_select[i]
            inputs = tokenizer(prompt, max_length=10000, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                try:
                    _ = model(**inputs)
                except:
                    print(i, "@@@-error-@@@", inputs.input_ids.size())

        assert len(hidden_states_lang) == num_layers
        hidden_states_lang_layers = {}
        for i_layers in range(num_layers):
            tmp = torch.cat(hidden_states_lang[i_layers], dim=0)
            assert tmp.size(0) > tokens_select
            hidden_states_lang_layers[i_layers] = tmp[:tokens_select]

        all_langs_hidden_states[lang] = hidden_states_lang_layers

    dif_layers_cos_sim = []
    for i_layer in range(num_layers):
        lang_a, lang_b = langs.split("_")
        tmp_a = all_langs_hidden_states[lang_a][i_layer].to(device)
        tmp_b = all_langs_hidden_states[lang_b][i_layer].to(device)
        assert tmp_a.size() == tmp_b.size()
        tmp_a_norm = tmp_a / tmp_a.norm(dim=1, keepdim=True)
        tmp_b_norm = tmp_b / tmp_b.norm(dim=1, keepdim=True)
        tmp_cos = torch.matmul(tmp_a_norm, tmp_b_norm.T)
        dif_layers_cos_sim.append(tmp_cos.mean().cpu())
        del tmp_a, tmp_b, tmp_a_norm, tmp_b_norm, tmp_cos

    for i_layer in range(num_layers):
        print(f"{dif_layers_cos_sim[i_layer]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.2-3B",
                        help="Base model name or path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing language JSONL files")
    parser.add_argument("--new_langs", type=str, default="el_hu_tr",
                        help="New languages (underscore-separated)")
    parser.add_argument("--old_langs", type=str, default="en_es_zh",
                        help="Existing languages (underscore-separated)")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to MoE adapter checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for computation")

    args = parser.parse_args()

    # Build data path dictionary from data_dir
    data_path_dict = {}
    all_langs = args.new_langs.split("_") + args.old_langs.split("_")
    for lang in all_langs:
        for pattern in [f"{lang}-llama-2B.jsonl", f"{lang}-qwen-2B.jsonl",
                       f"{lang}-llama-50K.jsonl", f"{lang}-qwen-50K.jsonl",
                       f"SlimPajama-50K.jsonl" if lang == "en" else None,
                       f"2023-14_zh_middle_0009-50K.jsonl" if lang == "zh" else None]:
            if pattern and os.path.exists(os.path.join(args.data_dir, pattern)):
                data_path_dict[lang] = os.path.join(args.data_dir, pattern)
                break

    print(f"Data paths: {data_path_dict}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    moe_config = PeftConfig.from_pretrained(args.adapter_path)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map=args.device)
    print("MoE config:", moe_config)
    model = PeftModel.from_pretrained(model,
                                        model_id=args.adapter_path,
                                        config=moe_config,
                                        is_trainable=True)

    new_langs = args.new_langs.split("_")
    old_langs = args.old_langs.split("_")

    for i in range(len(new_langs)):
        for j in range(i+1, len(new_langs)):
            cur_langs = new_langs[i] + "_" + new_langs[j]
            print("+++++", cur_langs)
            main(model, tokenizer, cur_langs, data_path_dict, device=args.device)

    for i in new_langs:
        for j in old_langs:
            cur_langs = i + "_" + j
            print("+++++", cur_langs)
            main(model, tokenizer, cur_langs, data_path_dict, device=args.device)
