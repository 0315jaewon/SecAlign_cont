import argparse
from copy import deepcopy

import test
from utils import form_llm_input, jload, load_vllm_model, test_model_output_vllm


def main():
    parser = argparse.ArgumentParser(description="Print davinci_003 examples and model outputs.")
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        default="/home/gcpuser/models/Llama-3.2-1B-Instruct_secalign_adapter",
    )
    parser.add_argument("--test_data", default="data/davinci_003_outputs.json")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--attack", default="none")
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--instruction_hierarchy", action="store_true", default=True)
    args = parser.parse_args()
    attack_func = getattr(test, args.attack)

    data = jload(args.test_data)
    examples = data[args.start : args.start + args.num_examples]

    model, tokenizer = load_vllm_model(args.model_name_or_path, args.tensor_parallel_size)
    prompts = form_llm_input(
        deepcopy(examples),
        attack_func,
        tokenizer.apply_chat_template,
        args.instruction_hierarchy,
        defense="none",
    )

    base_model_path = args.model_name_or_path.split("_")[0]
    adapter_path = args.model_name_or_path if base_model_path != args.model_name_or_path else None
    outputs = test_model_output_vllm(
        prompts,
        model,
        tokenizer,
        adapter_path,
        args.lora_alpha,
    )

    for i, (example, prompt, output) in enumerate(zip(examples, prompts, outputs), start=args.start):
        print("=" * 100)
        print(f"EXAMPLE {i}")
        print("-" * 100)
        print("ATTACK:")
        print(args.attack)
        print("-" * 100)
        print("INSTRUCTION:")
        print(example["instruction"])
        print("-" * 100)
        print("INPUT:")
        print(example["input"])
        print("-" * 100)
        print("REFERENCE OUTPUT:")
        print(example["output"])
        print("-" * 100)
        print("MODEL OUTPUT:")
        print(output)
        print("-" * 100)
        print("FORMATTED PROMPT:")
        print(prompt)


if __name__ == "__main__":
    main()
