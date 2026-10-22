import argparse
import os

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from transformers import AutoTokenizer

from specwire.orchestrator import generate
from specwire.rpc import TOKEN_ENV, DraftClient, TargetClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--draft", default="localhost:50051")
    parser.add_argument("--target", default="localhost:50052")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-temperature", type=float)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-steps", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tokenizer(args.prompt)["input_ids"]
    console = Console(highlight=False)
    accepted = []

    def quoted(tokens):
        return escape('"' + tokenizer.decode(tokens).replace("\n", "\\n") + '"')

    def show_round(draft, n, nxt, target_choices, draft_s, verify_s):
        accepted.append(n)
        console.print(
            f"round {len(accepted)}   draft {draft_s * 1000:.0f}ms   verify {verify_s * 1000:.0f}ms"
        )
        table = Table("spot", "draft", "target top")
        for i, token in enumerate(draft):
            style = "green" if i < n else "red" if i == n else "dim"
            table.add_row(str(i + 1), quoted([token]), quoted([target_choices[i]]), style=style)
        if n == len(draft):
            table.add_row(str(n + 1), "", quoted([target_choices[n]]), style="yellow")
        console.print(table)
        source = "1 bonus" if n == len(draft) else "1 from target"
        console.print(f"kept: {quoted(draft[:n] + [nxt])}  ({n} accepted + {source})")

    output = generate(
        DraftClient(args.draft, os.environ.get(TOKEN_ENV)),
        TargetClient(args.target, os.environ.get(TOKEN_ENV)),
        ids,
        args.max_new_tokens,
        args.k,
        tokenizer.eos_token_id,
        show_round if args.show_steps else None,
        args.temperature,
        args.draft_temperature,
        args.top_k,
        args.seed,
    )
    if args.show_steps:
        rounds = len(accepted)
        console.print(
            f"{rounds} rounds   {len(output)} tokens   "
            f"avg accepted {sum(accepted) / rounds:.2f}/{args.k}   "
            f"{len(output) / rounds:.2f} tokens per target call"
        )
    print(args.prompt + tokenizer.decode(output))
