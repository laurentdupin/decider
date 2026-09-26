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
