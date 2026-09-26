"""Generate Python-reference tokens, slots, logits, probabilities, and response.

The resulting JSON can be compared directly with decider_native_canary
--diagnostics after exporting the same checkpoint to unquantized GGUF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from decider import systemone as system_one
from decider import temperature
from decider.model import DecisionModel
from decider.prompt import chat_for, load_decider_config
from decider.prompt_fast import build_rows


DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Python Decider parity fixture for the native harness."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--max-state-tokens", type=int, default=32768)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    state = system_one.render_state(request["state"])
    questions = {
        key: system_one.render_question(value)
        for key, value in request.get("questions", {}).items()
    }
    independent = request.get("independent", True)
    config = load_decider_config(args.checkpoint)
    isolated = bool(config.get("isolated_levels", False)) and independent

    model = DecisionModel(args.checkpoint, dtype=DTYPES[args.dtype], grad_ckpt=False)
    model.to(args.device).eval()
    flat, index = system_one.plan_rows(questions, isolated)
    pairs = [(row["question"], list(row["options"])) for row in flat]
    rows = [[pair] for pair in pairs] if independent else [pairs]
    items, _ = build_rows(
        model.tok,
        state,
        rows,
        max_ctx_tokens=args.max_state_tokens,
        chat=chat_for(model.tok, config),
    )
    row_types = system_one.row_types(questions, index)
    grouped_types = [[item] for item in row_types] if independent else [row_types]
    base_temperature, by_type = temperature.from_config(config)[0]
    effective = temperature.effective(base_temperature, by_type)

    all_probabilities: list[list[float]] = []
    diagnostic_rows = []
    with torch.no_grad():
        for item, types in zip(items, grouped_types):
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=args.device)
            attention = torch.ones_like(ids)
            slots = torch.tensor(item["slots"], dtype=torch.long, device=args.device)
            slot_batch = torch.zeros(len(item["slots"]), dtype=torch.long, device=args.device)
            nopts = torch.tensor(item["nopts"], dtype=torch.long, device=args.device)
            logits = model.slot_logits(ids, attention, slots, slot_batch, nopts)
            selected_logits = []
            probabilities = []
            for slot, answer_type in enumerate(types):
                count = item["nopts"][slot]
                selected = logits[slot, :count].float().cpu()
                probs = torch.softmax(selected / effective[answer_type], dim=-1)
                selected_logits.append(selected.tolist())
                probabilities.append(probs.tolist())
                all_probabilities.append(probs.tolist())
            diagnostic_rows.append(
                {
                    "token_ids": item["ids"],
                    "slots": item["slots"],
                    "option_counts": item["nopts"],
                    "types": types,
                    "selected_logits": selected_logits,
                    "probabilities": probabilities,
                }
            )

    result = {
        "checkpoint": args.checkpoint,
        "dtype": args.dtype,
        "label_token_ids": model.letters.cpu().tolist(),
        "rows": diagnostic_rows,
        "response": {
            "model": "decider-" + str(config.get("version", "dev")),
            "answers": system_one.assemble(questions, index, all_probabilities),
            "usage": {
                "input_tokens": system_one.unique_tokens(items),
                "output_tokens": 0,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
