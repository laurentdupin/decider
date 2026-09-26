"""Run JevBench's public tasks through the persistent InferBridge harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[3]
JEVBENCH_ROOT = ROOT / "third_party" / "jevbench"
sys.path.insert(0, str(JEVBENCH_ROOT))

# JevBench's durable ledger uses POSIX flock. This runner is deliberately a
# single Windows process, so a no-op compatibility module preserves the same
# append/settle semantics without pretending to provide cross-process locking.
try:
    import fcntl  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_SH = 1
    fcntl.LOCK_EX = 2
    fcntl.flock = lambda _file, _operation: None
    sys.modules["fcntl"] = fcntl

from jevbench.adapters.base import DecisionResult  # noqa: E402
from jevbench.budget import Ledger  # noqa: E402
from jevbench.runner import Runner  # noqa: E402
from jevbench.summarize import public_export, summarize  # noqa: E402
from jevbench.tasks import dataset_hash, load_jsonl  # noqa: E402


class InferBridgeAdapter:
    name = "inferbridge_decider"
    cost_basis = "self_hosted_local_unpriced"
    price_input_per_m = None
    price_output_per_m = None

    def __init__(self, executable: Path, model: Path, backend: str,
                 model_parameters: dict, gpu: str | None):
        self.executable = executable.resolve()
        self.model_path = model.resolve()
        self.backend = backend
        self.model = model.name
        self.parameters = model_parameters
        self.gpu = gpu
        self.process: subprocess.Popen[str] | None = None

    def load(self) -> None:
        if self.process is not None:
            return
        environment = os.environ.copy()
        if self.gpu is not None:
            environment["GGML_VK_VISIBLE_DEVICES"] = self.gpu
        command = [
            str(self.executable), str(self.model_path), self.backend,
            "--model-parameters", json.dumps(self.parameters, separators=(",", ":")),
            "--ready",
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
            env=environment,
        )
        ready = self.process.stdout.readline() if self.process.stdout else ""
        if ready.strip() != '{"ready":true}':
            error = self.process.stderr.read() if self.process.stderr else ""
            self.close()
            raise RuntimeError(f"InferBridge JSONL startup failed: {error.strip()}")

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)

    @staticmethod
    def build_request(task) -> dict:
        question = {
            "type": task.question["type"],
            "instructions": task.question["instructions"],
        }
        if task.question.get("criteria") is not None:
            question["criteria"] = task.question["criteria"]
        return {
            "state": task.state,
            "questions": {"decision": question},
            "independent": True,
        }

    def run(self, task) -> DecisionResult:
        self.load()
        body = self.build_request(task)
        result = DecisionResult(
            adapter=self.name, ok=False, probs_source="native",
            model=self.model, request_body=body,
        )
        assert self.process is not None
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        started = time.perf_counter()
        try:
            self.process.stdin.write(json.dumps(body, ensure_ascii=False,
                                                separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                error = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"InferBridge JSONL process exited: {error.strip()}")
            response = json.loads(line)
        except Exception as error:  # benchmark failures are recorded, not hidden
            result.latency_s = time.perf_counter() - started
            result.error = f"{type(error).__name__}: {str(error)[:300]}"
            return result
        result.latency_s = time.perf_counter() - started
        result.raw = response
        result.model = response.get("model") or result.model
        result.usage = response.get("usage") or {}
        answer = (response.get("answers") or {}).get("decision")
        if not isinstance(answer, dict) or answer.get("type") != task.question["type"]:
            result.error = "native answer type mismatch"
            return result
        try:
            if task.question["type"] == "noul":
                probability = float(answer["noul"])
                result.probs = {"yes": probability, "no": 1.0 - probability}
            else:
                probabilities = answer["probabilities"]
                if not isinstance(probabilities, dict):
                    raise TypeError("probabilities must be an object")
                result.probs = {str(key): float(value)
                                for key, value in probabilities.items()}
        except (KeyError, TypeError, ValueError) as error:
            result.error = f"answer parse failed: {error}"
            return result
        result.ok = True
        return result

    def reserve_estimate(self, task) -> float:
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=ROOT / "out" /
                        "decider-native-vulkan" / "Release" /
                        "decider_native_jsonl.exe")
    parser.add_argument("--backend", choices=("CPU", "VULKAN"), default="VULKAN")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_paths = [
        JEVBENCH_ROOT / "datasets" / "public" / name
        for name in ("easy.jsonl", "original.jsonl", "hard.jsonl")
    ]
    tasks = [task for path in task_paths for task in load_jsonl(str(path))]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)

    parameters = {
        "context_size": args.context_size,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "disable_vision": True,
    }
    adapter = InferBridgeAdapter(args.executable, args.model, args.backend,
                                 parameters, args.gpu)
    load_started = time.perf_counter()
    adapter.load()
    load_seconds = time.perf_counter() - load_started
    print(f"[jevbench] model loaded in {load_seconds:.2f}s", flush=True)
    try:
        runner = Runner(
            adapter,
            Ledger(str(args.output_dir / "ledger.jsonl"), cap_usd=0.0),
            raw_dir=str(args.output_dir / "raw"),
            default_reserve_usd=0.0,
        )
        records = runner.run_all(
            tasks, results_path=str(args.output_dir / "results.jsonl"))
    finally:
        adapter.close()

    summary = summarize(tasks, records, 0.0, headline_only=True)
    export = public_export(summary, tasks, records)
    export["dataset_hash"] = dataset_hash(tasks)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "public-export.json").write_text(
        json.dumps(export, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")
    manifest = {
        "adapter": adapter.name,
        "backend": args.backend,
        "gpu": args.gpu,
        "model": str(args.model.resolve()),
        "model_parameters": parameters,
        "model_load_seconds": load_seconds,
        "dataset_hash": dataset_hash(tasks),
        "task_files": [str(path.resolve()) for path in task_paths],
        "tasks": len(tasks),
        "attempted": len(records),
        "note": "Stock Qwen runtime probe; not Decider-fine-tuned or calibrated.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if len(records) == len(tasks) else 3


if __name__ == "__main__":
    raise SystemExit(main())
