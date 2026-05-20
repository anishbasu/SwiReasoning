import os
import re
import torch
import torch.nn.functional as F
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import transformers
        transformers.set_seed(seed)
    except Exception:
        pass


def apply_sampling_filter(logits, top_k=0, top_p=1.0, min_p=0.0):
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, float('-inf'), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        logits = torch.where(probs < min_p, float('-inf'), logits)
    return logits


def _select_tensor_batch(value, indices):
    if value is None or not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    return value.index_select(0, indices.to(value.device))


def _batch_select_cache_layer(layer, indices):
    has_linear_states = hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")
    if has_linear_states and hasattr(layer, "reorder_cache"):
        layer.reorder_cache(indices)
    elif hasattr(layer, "batch_select_indices"):
        layer.batch_select_indices(indices)
    elif hasattr(layer, "reorder_cache"):
        layer.reorder_cache(indices)
    else:
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(layer, attr, _select_tensor_batch(value, indices))

    if hasattr(layer, "max_batch_size"):
        try:
            layer.max_batch_size = int(indices.numel())
        except Exception:
            pass


def _needs_layerwise_batch_select(past_key_values):
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return False
    for layer in layers:
        has_linear_states = hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")
        if has_linear_states or not hasattr(layer, "batch_select_indices"):
            return True
    return False


def batch_select_hybrid_cache(past_key_values, indices):
    if past_key_values is None:
        return past_key_values

    if hasattr(past_key_values, "batch_select_indices") and not _needs_layerwise_batch_select(past_key_values):
        past_key_values.batch_select_indices(indices)
        return past_key_values

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            _batch_select_cache_layer(layer, indices)
        return past_key_values

    if hasattr(past_key_values, "batch_select_indices"):
        past_key_values.batch_select_indices(indices)
        return past_key_values

    if isinstance(past_key_values, tuple):
        selected_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                selected_layers.append(tuple(_select_tensor_batch(v, indices) for v in layer))
            else:
                selected_layers.append(_select_tensor_batch(layer, indices))
        return tuple(selected_layers)
    return past_key_values


def get_math_symbols_ids(tokenizer):
    math_symbols = [
        "+", "-", "*", "/", "^", "=", "<", ">", "\\leq", "\\geq", "\\neq", "\\approx", "\\sim", "\\equiv", "\\to", "\\implies", "\\iff",
        "(", ")", "[", "]", "{", "}", "\\left(", "\\right)", "\\left[", "\\right]", "\\left\\{", "\\right\\}",
        "\\begin{pmatrix}", "\\end{pmatrix}",
        "\\frac", "\\dfrac", "\\sqrt", "\\sqrt[]",
        "\\in", "\\notin", "\\subset", "\\supset", "\\subseteq", "\\supseteq", "\\cup", "\\cap", "\\emptyset", "\\varnothing",
        "\\pi", "\\theta", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\lambda", "\\mu", "\\nu",
        "\\sin", "\\cos", "\\tan", "\\arcsin", "\\arccos", "\\arctan", "\\log", "\\ln", "\\exp",
        "_", "\\binom", "\\choose", "\\cdot", "\\dots", "\\ldots", "\\cdots", "\\vdots", "\\ddots",
        "\\mathbb", "\\mathbf", "\\mathrm", "\\text", "\\mbox",
        "\\infty", "\\circ", "\\prime", "\\ast", "\\star", "\\triangle", "\\triangleleft", "\\triangleright", "\\perp", "\\parallel", "\\angle",
        "\\boxed", "\\overline", "\\underline", "\\lceil", "\\rceil", "\\lfloor", "\\rfloor", "\\left", "\\right", "\\mid", "|", "\\vert", "\\Vert",
        "\\because", "\\therefore", "\\forall", "\\exists", "\\wedge", "\\vee", "\\neg",
        "\\sum", "\\prod", "\\int", "\\lim", "\\min", "\\max", "\\arg", "\\deg", "\\gcd", "\\operatorname",
        "\\cot", 
        "\\cotg", "\\sec", "\\csc",
    ]
    math_symbols += [chr(c) for c in range(ord('0'), ord('9')+1)]
    math_symbols += [chr(c) for c in range(ord('a'), ord('z')+1)]
    math_symbols += [chr(c) for c in range(ord('A'), ord('Z')+1)]
    math_token_ids = set()
    for symbol in math_symbols:
        math_token_ids.update(tokenizer.encode(symbol, add_special_tokens=False))
    return math_token_ids
    

def generate_cot(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)

    stream_callback = kwargs.pop("stream_callback", None)

    # ============================================

    batch_size = input_ids.shape[0]
    device = input_ids.device

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))

    generated = input_ids.clone()
    attn_mask = attention_mask.clone()
    past_key_values = None
        
    for step in range(max_new_tokens):
        cur_batch = generated.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {"input_ids": generated, "attention_mask": attn_mask}
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)
            attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1)
            model_inputs = {"input_ids": next_tokens.unsqueeze(1), "past_key_values": past_key_values, "attention_mask": attn_mask}

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]  # [cur_batch, vocab]
        logits = next_token_logits / temperature
        logits = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)

        probs = F.softmax(logits, dim=-1)
        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])

        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

        if len(unfinished_idx) == 0:
            break
        generated = generated[keep_idx]
        next_tokens = next_tokens[keep_idx]
        attention_mask = attention_mask[keep_idx]
        attn_mask = attn_mask[keep_idx]
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=generated.device)
        past_key_values = batch_select_hybrid_cache(past_key_values, keep_idx_tensor)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


def generate_swir(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)

    # ---- swir ----
    alpha_0                = kwargs.pop("alpha_0", 1.0) # adjustable
    beta_0                 = kwargs.pop("beta_0", 0.7)
    window_size            = kwargs.pop("window_size", 512)
    thinking_token_id      = kwargs.pop("thinking_token_id", None)
    end_thinking_token_id  = kwargs.pop("end_thinking_token_id", None)
    max_switch_count       = kwargs.pop("max_switch_count", None) # adjustable for efficiency
    math_ids_tensor        = kwargs.pop("math_ids_tensor", None)
    convergence_words      = kwargs.get("convergence_words", "</think>")
    termination_words      = kwargs.get("termination_words", "</think>\n\nThe final answer is")
    termination_max_tokens = kwargs.pop("termination_max_tokens", 32)

    stream_callback       = kwargs.pop("stream_callback", None)

    # ============================================

    batch_size, device = input_ids.shape[0], input_ids.device
    E = model.get_input_embeddings().weight  # [vocab_size, dim]
    if thinking_token_id is None or end_thinking_token_id is None:
        thinking_token_id = tokenizer.convert_tokens_to_ids("<think>")
        end_thinking_token_id = tokenizer.convert_tokens_to_ids("</think>")
    start_thinking_emb, end_thinking_emb = E[thinking_token_id], E[end_thinking_token_id]
    line_break_emb = E[tokenizer.convert_tokens_to_ids("\\n")]
    past_key_values = None
        
    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size)) # bs >= 1 is supported
    mode = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: soft, 1: normal
    mode_stay_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    locked_normal_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    if max_switch_count is not None:
        switch_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        convergence_ids = tokenizer.encode(convergence_words, add_special_tokens=False)
        termination_ids = tokenizer.encode(termination_words, add_special_tokens=False)
        injecting = torch.zeros(batch_size, dtype=torch.bool, device=device)
        inject_queues = [[] for _ in range(batch_size)]
        answer_budget = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        cur_batch = attention_mask.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {
                "input_ids": input_ids.clone(), 
                "attention_mask": attention_mask,
            }
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "inputs_embeds": last_emb.unsqueeze(1), 
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        
        logits_original = outputs.logits[:, -1, :]
        probs_original = F.softmax(logits_original, dim=-1)
        logits = logits_original / temperature  
        logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)  # [B, N, V]
        probs = F.softmax(logits_filtered, dim=-1)

        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)  # [B, N]
        locked_normal_mask = locked_normal_mask | (next_tokens == end_thinking_token_id)

        if max_switch_count is not None and injecting.any():
            mask_list = [injecting[i].item() and len(inject_queues[i]) > 0 for i in range(cur_batch)]
            force_mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
            if force_mask.any():
                force_toks = torch.tensor([inject_queues[i].pop(0) for i in range(cur_batch) if mask_list[i]], \
                                          device=device, dtype=torch.long)
                next_tokens[force_mask] = force_toks
            if injecting.any():
                done_mask = torch.tensor([injecting[i] and (len(inject_queues[i]) == 0) for i in range(cur_batch)], \
                                         device=device, dtype=torch.bool)
                injecting[done_mask] = False
        
        cur_entropy = -(probs_original * (probs_original.clamp(min=1e-12).log())).sum(dim=-1)
        if step == 0:
            cur_ref_entropy = cur_entropy.clone()
        else:
            mode_stay_steps += 1
            allow_switch = (mode_stay_steps >= window_size)
            to_normal = (mode == 0) & (cur_entropy < cur_ref_entropy)
            to_soft = (mode == 1) & (cur_entropy > cur_ref_entropy) & allow_switch & (~locked_normal_mask)
            mode[to_normal] = 1
            mode[to_soft] = 0
            mode_stay_steps[to_normal | to_soft] = 0
            cur_ref_entropy[to_normal | to_soft] = cur_entropy[to_normal | to_soft]
            if max_switch_count is not None:
                switch_count = switch_count + to_normal.long() 
            
        is_normal = (mode == 1) | locked_normal_mask
        if math_ids_tensor is not None:
            is_math_token = (next_tokens.unsqueeze(-1) == math_ids_tensor).any(dim=-1)
            is_normal[is_math_token] = True
        is_soft = ~is_normal
        
        normal_emb = E[next_tokens]
        soft_emb = torch.matmul(probs_original, E)

        alpha = alpha_0 + (1 - alpha_0) * float(step) / float(max_new_tokens)
        if step == 0:
            soft_emb = 0.9 * soft_emb + 0.1 * line_break_emb
        else:
            mixed_emb = alpha * soft_emb + (1 - alpha) * start_thinking_emb
            soft_emb = torch.where(to_soft[:, None], mixed_emb, soft_emb)
        beta = beta_0 + (1 - beta_0) * float(step) / float(max_new_tokens)
        if step > 0:
            mixed_emb = beta * soft_emb + (1 - beta) * end_thinking_emb
            normal_emb = torch.where(to_normal[:, None], mixed_emb, normal_emb)
        last_emb = torch.where(is_soft[:, None], soft_emb, normal_emb)

        if max_switch_count is not None and step > 0:
            trigger = (switch_count >= max_switch_count) & (switch_count <= 2 * max_switch_count) & to_normal
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(convergence_ids)
                injecting = injecting | trigger

            trigger = (switch_count > 2 * max_switch_count) & to_normal
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(termination_ids) 
                injecting = injecting | trigger 
                answer_budget[trigger] = termination_max_tokens
            active = (answer_budget >= 0)
            if active.any():
                answer_budget = torch.where(active, answer_budget - 1, answer_budget)

        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])
        
        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        if max_switch_count is not None:
            budget_done = (answer_budget == 0) 
            cur_finished = cur_finished | budget_done

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
        if len(unfinished_idx) == 0:
            break
        last_emb = last_emb[keep_idx]
        attention_mask = attention_mask[keep_idx]
        mode = mode[keep_idx]
        mode_stay_steps = mode_stay_steps[keep_idx]
        cur_ref_entropy = cur_ref_entropy[keep_idx]
        locked_normal_mask = locked_normal_mask[keep_idx]
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
        past_key_values = batch_select_hybrid_cache(past_key_values, keep_idx_tensor)
        if max_switch_count is not None:
            switch_count = switch_count[keep_idx]
            injecting = injecting[keep_idx]
            inject_queues = [inject_queues[i] for i in keep_idx.tolist()]
            answer_budget = answer_budget[keep_idx]

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out
