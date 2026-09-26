# InferBridge Decider harness

This directory contains the native InferBridge 2 harness for Decider. The
shipping interface is one UTF-8 JSON input and one UTF-8 JSON output using the
same request and response shapes as `POST /v1/systemone`.

The implementation is deliberately Decider-owned. It uses the standalone
Qwen GGUF harness as an ABI/lifecycle reference, while keeping Decider prompt
construction, typed-answer assembly, calibration, and compatibility tests in
this repository.

## Current scope

- C++ SystemOne request validation and row planning.
- Python-compatible state rendering, including long-array index annotations.
- Choice, Score, Noul, and isolated-Score answer assembly.
- Plain-layout narrow prompt blocks with the same tokenization boundary as the
  Python implementation.
- A pinned llama.cpp dependency capable of loading dense and MoE Qwen3.5 GGUF
  models and returning logits at requested prompt positions.
- An InferBridge ABI 2 harness with CPU/Vulkan runtime selection, GGUF model
  loading, asynchronous submission, cancellation, and bounded JSON output.
- Model-backed plain/state-first inference for independent rows with 2..10
  options, including per-answer-type temperatures from `decider_config.json`.

Chat metadata, wide labels, packed requests, and verified shared-prefix state
copying are explicit follow-up gates; the harness rejects an unsupported model
configuration or request rather than silently using a different prompt.

## Configure and test the protocol layer

```powershell
cmake -S native/inferbridge_decider -B out/decider-native-protocol `
  -DDECIDER_BUILD_HARNESS=OFF -DDECIDER_BUILD_TESTS=ON
cmake --build out/decider-native-protocol --config Release
ctest --test-dir out/decider-native-protocol -C Release --output-on-failure
```

Configure the CPU harness and its ABI smoke test with:

```powershell
cmake -S native/inferbridge_decider -B out/decider-native `
  -DDECIDER_BUILD_HARNESS=ON -DDECIDER_BUILD_TESTS=ON `
  -DDECIDER_LLAMA_VULKAN=OFF
cmake --build out/decider-native --config Release
ctest --test-dir out/decider-native -C Release --output-on-failure
```

`third_party/llama.cpp` is pinned to the revision validated by the existing
Qwen GGUF harness. Do not advance it without compiling the ABI and rerunning
prompt/logit parity fixtures.

## Export a checkpoint

Install the pinned converter requirements in a dedicated environment, then
export a local Hugging Face checkpoint. The exporter refuses layouts the
current harness cannot reproduce and packages `decider_config.json` beside the
GGUF model.

```powershell
python -m venv .venv-gguf
.venv-gguf\Scripts\python -m pip install `
  -r third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
.venv-gguf\Scripts\python native/inferbridge_decider/tools/export_gguf.py `
  C:\models\decider-2b C:\models\decider-2b-native\decider-2b-bf16.gguf
```

Use BF16 for the initial Python/native logit comparison. Quantization changes
the probability distribution and must pass its own evaluation and temperature
calibration before being published as an interchangeable Decider build.

After export, exercise the real ABI, model loader, tokenizer, logits path, and
JSON response writer with the canary executable:

```powershell
out\decider-native\Release\decider_native_canary.exe `
  C:\models\decider-2b-native\decider-2b-bf16.gguf `
  native\inferbridge_decider\examples\request.json CPU
```

The checkpoint's `decider_config.json` must remain beside the GGUF file. Use
`VULKAN` as the final argument only in a build configured with
`-DDECIDER_LLAMA_VULKAN=ON`; the harness now fails early when no Vulkan GPU
backend is actually registered.
