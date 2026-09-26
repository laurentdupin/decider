"""Export a local Decider Hugging Face checkpoint for the native harness.

Run this from a Python environment containing the pinned llama.cpp conversion
requirements. Python is used only for this offline conversion step; the
shipping InferBridge runtime remains native.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LLAMA = ROOT / "third_party" / "llama.cpp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a local Decider checkpoint to a native-harness GGUF package."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="Output .gguf path")
    parser.add_argument(
        "--outtype",
        choices=("auto", "bf16", "f16", "f32", "q8_0"),
        default="bf16",
        help="Start parity work with bf16; quantized builds require separate calibration.",
    )
    parser.add_argument("--use-temp-file", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    config_path = checkpoint / "decider_config.json"
    if not checkpoint.is_dir() or not config_path.is_file():
        parser.error("checkpoint must be a local directory containing decider_config.json")
    if output.suffix.lower() != ".gguf":
        parser.error("output must have a .gguf extension")
    if not (LLAMA / "convert_hf_to_gguf.py").is_file():
        parser.error("third_party/llama.cpp is not initialized")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    layout = config.get("layout", "chat" if config.get("chat_template") is True else "plain")
    if layout != "plain":
        parser.error(
            f"native harness milestone 1 supports only layout='plain'; checkpoint declares {layout!r}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(LLAMA / "convert_hf_to_gguf.py"),
        str(checkpoint),
        "--outfile",
        str(output),
        "--outtype",
        args.outtype,
    ]
    if args.use_temp_file:
        command.append("--use-temp-file")
    subprocess.run(command, cwd=LLAMA, check=True)

    packaged_config = output.parent / "decider_config.json"
    if config_path != packaged_config:
        shutil.copy2(config_path, packaged_config)
    provenance = {
        "schema_version": 1,
        "model_file": output.name,
        "outtype": args.outtype,
        "source_config_sha256": sha256(config_path),
        "llama_cpp_commit": subprocess.check_output(
            ["git", "-C", str(LLAMA), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (output.parent / "inferbridge-decider-export.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {output}")
    print(f"Packaged {packaged_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
