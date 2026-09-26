"""Generate a Transformers reference fixture for image-plus-text Decider inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from decider import systemone as system_one
from decider import temperature
from decider.prompt import chat_for, label_table, load_decider_config
from decider.prompt_fast import build_rows
from decider.vision.model import IMG


DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}


def find_subsequence(sequence: list[int], needle: list[int]) -> int:
    for offset in range(len(sequence) - len(needle) + 1):
        if sequence[offset : offset + len(needle)] == needle:
            return offset
    raise RuntimeError("processor output does not contain the Decider text tokens")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Python vision-Decider parity fixture."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("request", type=Path)
    parser.add_argument("image", type=Path)
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
    flat, index = system_one.plan_rows(questions, isolated)
    pairs = [(row["question"], list(row["options"])) for row in flat]
    rows = [[pair] for pair in pairs] if independent else [pairs]

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    tokenizer = processor.tokenizer
    items, _ = build_rows(
        tokenizer,
        state,
        rows,
        max_ctx_tokens=args.max_state_tokens,
        chat=chat_for(tokenizer, config),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint, dtype=DTYPES[args.dtype]
    ).to(args.device).eval()
    _, label_ids, _ = label_table(tokenizer)
    labels = torch.tensor(label_ids, dtype=torch.long, device=args.device)
    row_types = system_one.row_types(questions, index)
    grouped_types = [[item] for item in row_types] if independent else [row_types]
    base_temperature, by_type = temperature.from_config(config)[0]
    effective = temperature.effective(base_temperature, by_type)
    image = Image.open(args.image).convert("RGB")

    all_probabilities: list[list[float]] = []
    diagnostic_rows = []
    input_tokens = 0
    with torch.no_grad():
        for item, types in zip(items, grouped_types):
            text = IMG + tokenizer.decode(item["ids"])
            inputs = processor(images=[image], text=[text], return_tensors="pt")
            processed_ids = inputs["input_ids"][0].tolist()
            text_offset = find_subsequence(processed_ids, item["ids"])
            processed_slots = torch.tensor(
                [text_offset + slot for slot in item["slots"]],
                dtype=torch.long,
                device=args.device,
            )
            forwarded = {
                key: value.to(args.device)
                for key, value in inputs.items()
                if key in (
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "image_grid_thw",
                    "mm_token_type_ids",
                )
            }
            hidden = model.model(**forwarded, use_cache=False).last_hidden_state
            slot_hidden = hidden[0, processed_slots]
            logits = F.linear(slot_hidden, model.lm_head.weight[labels]).float()
            selected_logits = []
            probabilities = []
            for slot, answer_type in enumerate(types):
                count = item["nopts"][slot]
                selected = logits[slot, :count].cpu()
                probs = torch.softmax(selected / effective[answer_type], dim=-1)
                selected_logits.append(selected.tolist())
                probabilities.append(probs.tolist())
                all_probabilities.append(probs.tolist())
            input_tokens += len(processed_ids)
            diagnostic_rows.append(
                {
                    "token_ids": item["ids"],
                    "slots": item["slots"],
                    "option_counts": item["nopts"],
                    "types": types,
                    "selected_logits": selected_logits,
                    "probabilities": probabilities,
                    "processed_input_tokens": len(processed_ids),
                    "image_grid_thw": inputs["image_grid_thw"][0].tolist(),
                }
            )

    result = {
        "checkpoint": args.checkpoint,
        "image": str(args.image),
        "dtype": args.dtype,
        "label_token_ids": label_ids,
        "rows": diagnostic_rows,
        "response": {
            "model": "decider-" + str(config.get("version", "dev")),
            "answers": system_one.assemble(questions, index, all_probabilities),
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
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
