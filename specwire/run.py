import argparse

from transformers import AutoTokenizer

from specwire.orchestrator import generate
from specwire.rpc import DraftClient, TargetClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--draft", default="localhost:50051")
    parser.add_argument("--target", default="localhost:50052")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tokenizer(args.prompt)["input_ids"]
    output = generate(
        DraftClient(args.draft),
        TargetClient(args.target),
        ids,
        args.max_new_tokens,
        args.k,
        tokenizer.eos_token_id,
    )
    print(args.prompt + tokenizer.decode(output))
