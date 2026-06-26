import os
import argparse
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from generation_utils import (
    set_seed,
    get_math_symbols_ids,
    generate_swir
)
from grader import answer_extraction


def build_chat_text(tokenizer, user_text: str) -> str:
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages = [{"role": "user", "content": user_text + suffix}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def main(args):
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"[Info] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map={"": local_rank}
    )
    device = model.device
    model.eval()
    math_symbols_ids = get_math_symbols_ids(tokenizer)
    math_ids_tensor = torch.tensor(list(math_symbols_ids), device=device)

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
        "gumbel_softmax_tau": args.gumbel_tau,
    }

    print(f"\n[Info] Model loaded on {device}")
    print("Type/Paste your question and press Enter.")
    print("Commands:")
    print("  exit or q -> [Exit]")
    print("  switch <N|none> -> [Set] swir max_switch_count = N (integer >= 1) or None (disabled)")
    print("  method <swir|cot|cot_greedy> -> [Set] generation method\n")

    while True:
        try:
            user_inp = input("User > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit]")
            break
        if user_inp == "":
            continue
        low = user_inp.lower()
        if low in ("exit", "q"):
            print("[Exit]")
            break
        if low.startswith("switch "):
            arg = user_inp.split(None, 1)[1].strip()
            if arg.lower() in ("none", "off"):
                args.max_switch_count = None
                print("[Info] swir max_switch_count set to None (disabled).")
            else:
                try:
                    v = int(arg)
                    if v < 1:
                        raise ValueError("must be >= 1")
                    args.max_switch_count = v
                    print(f"[Info] swir max_switch_count set to {v}.")
                except Exception as e:
                    print(f"[Warn] invalid value for switch: {arg} ({e})")
            continue
        if low.startswith("method "):
            arg = user_inp.split(None, 1)[1].strip().lower()
            if arg in ("swir", "cot", "cot_greedy"):
                args.method = arg
                print(f"[Info] method set to {args.method}.")
            else:
                print(f"[Warn] invalid method: {arg} (use swir|cot|cot_greedy)")
            continue

        text = build_chat_text(tokenizer, user_inp)
        model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
        print(f"{args.model_name} > ", end="", flush=True)
        
        content = ""
        num_tokens = 0
        is_thinking = False
        if args.method in ["cot", "cot_greedy"]:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            if args.method == "cot_greedy":
                gen_kwargs["do_sample"] = False
            gen_args = dict(
                **model_inputs,
                streamer=streamer,
                **gen_kwargs,
            )
            thread = Thread(target=model.generate, kwargs=gen_args)
            thread.start()
            for new_text in streamer:
                print(new_text, end="", flush=True)
                content += new_text
            thread.join()
            print("")
        else:
            def _stream_cb(new_ids: str):
                nonlocal content
                nonlocal num_tokens
                nonlocal is_thinking
                new_text = tokenizer.decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                content += new_text
                if "<think>" in new_text:
                    is_thinking = True
                if "</think>" in new_text:
                    is_thinking = False
                if is_thinking:
                    num_tokens += 1
                print(new_text, end="", flush=True)
            model_inputs["alpha_0"] = args.alpha
            model_inputs["max_switch_count"] = args.max_switch_count
            model_inputs["math_ids_tensor"] = math_ids_tensor
            model_inputs["convergence_words"] = "</think>" if "Qwen" in args.model_name else "\n\n</think>\n\n"
            model_inputs["stream_callback"] = _stream_cb
            _ = generate_swir(
                model,
                tokenizer,
                **model_inputs,   
                **gen_kwargs,   
            )
            print("") 
        
        if args.display_final_answer:
            _, answer = answer_extraction(content)
            RESET, BOLD, FG_W, BG_M = "\033[0m", "\033[1m", "\033[97m", "\033[45m" 
            def display_final_answer(text):
                print()
                print(f"{BOLD}{FG_W}{BG_M}[Final Answer]: {text}, [Num Tokens]: {num_tokens}{RESET}")
                print()
            display_final_answer(answer)

    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gumbel_tau", type=float, default=0.0)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--do_sample", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_new_tokens', type=int, default=38912)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display_final_answer", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--method", type=str, default="swir", choices=["swir", "cot", "cot_greedy"])
    parser.add_argument("--alpha", type=float, default=1.0) # swir-specific
    parser.add_argument("--max_switch_count", type=int, default=2) # swir-specific
    args = parser.parse_args()
    main(args)
