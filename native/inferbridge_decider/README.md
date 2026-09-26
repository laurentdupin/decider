# InferBridge Decider harness

This directory contains the native InferBridge 2 harness for Decider. The
shipping interface is one UTF-8 JSON input and one UTF-8 JSON output using the
same request and response shapes as `POST /v1/systemone`. A packaged vision
checkpoint additionally exposes a host RGBA8/BGRA8 image input.

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
- Model-backed plain/state-first inference for independent and packed rows with
  2..255 options, including exact wide-label token insertion and per-slot
  answer-type temperatures from `decider_config.json`.
- Qwen3.5-VL image-plus-text decisions through llama.cpp `mtmd`, with the
  language GGUF and vision projector loaded as one InferBridge model.

Chat metadata and verified shared-prefix state copying are explicit follow-up
gates; the harness rejects an unsupported model configuration or request
rather than silently using a different prompt.

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

Configure a Vulkan harness with the same ABI by enabling the llama.cpp Vulkan
backend. On Windows, `VULKAN_SDK` must name an installed SDK before configuring:

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.350.0'
cmake -S native/inferbridge_decider -B out/decider-native-vulkan `
  -DDECIDER_BUILD_HARNESS=ON -DDECIDER_BUILD_TESTS=ON `
  -DDECIDER_BUILD_CANARY=ON -DDECIDER_LLAMA_VULKAN=ON
cmake --build out/decider-native-vulkan --config Release --parallel
ctest --test-dir out/decider-native-vulkan -C Release --output-on-failure
```

Selecting `VULKAN` in the InferBridge runtime request makes the harness require
a registered Vulkan GPU and sets `n_gpu_layers=-1`, so all model layers are
offloaded. It does not silently fall back to CPU when Vulkan is unavailable.

`third_party/llama.cpp` is pinned to the revision validated by the existing
Qwen GGUF harness. Do not advance it without compiling the ABI and rerunning
prompt/logit parity fixtures.

## Export a checkpoint

Install the pinned converter requirements in a dedicated environment, then
export a local Hugging Face checkpoint. The exporter packages
`decider_config.json`, tokenizer-derived label IDs, model provenance, and—when
declared—exact chat head/tail text and tokens beside the GGUF model. The native
loader validates the packaged label table against the GGUF tokenizer.

The exporter excludes Qwen NextN speculative-draft metadata and tensors. They
are not used by Decider inference, and advertising a missing NextN layer causes
the pinned llama.cpp loader to expect draft-head tensors that are absent from
the released checkpoint.

```powershell
python -m venv .venv-gguf
.venv-gguf\Scripts\python -m pip install `
  -r third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
.venv-gguf\Scripts\python native/inferbridge_decider/tools/export_gguf.py `
  C:\models\decider-2b C:\models\decider-2b-native\decider-2b-bf16.gguf
```

For a multimodal checkpoint, export the projector at the same time. The
projector path is recorded in `inferbridge-decider-export.json`; the native
loader uses its presence to switch from one input port to two.

```powershell
.venv-gguf\Scripts\python native/inferbridge_decider/tools/export_gguf.py `
  C:\models\decider-2b-vision `
  C:\models\decider-2b-vision-native\model-decider-2b-vision-bf16.gguf `
  --mmproj-output `
  C:\models\decider-2b-vision-native\mmproj-decider-2b-vision-bf16.gguf
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

Append `--diagnostics` to include exact prompt token IDs, answer-slot positions,
the 255 label token IDs, and raw selected logits in the canary response. This
output is intended for Python/native parity fixtures and is disabled for normal
InferBridge submissions.
Use `--output out\parity\native.json` to write the canary response directly to
a comparison fixture.

For a packaged vision model, pass a PNG, JPEG, BMP, or other stb-supported
image to the canary. The canary decodes it to RGBA8 and submits the JSON and
image as separate InferBridge bindings:

```powershell
out\decider-native-vulkan\Release\decider_native_canary.exe `
  C:\models\decider-2b-vision-native\model-decider-2b-vision-bf16.gguf `
  native\inferbridge_decider\examples\simple_vision_choice.json `
  VULKAN --image media\cover.png --timing
```

The first native vision milestone accepts host images; the language model and
vision encoder both execute on Vulkan when the runtime backend is `VULKAN`.
Direct `IBRH_NATIVE_HANDLE_VULKAN_IMAGE` ingestion is not advertised yet, so
InferBridge stages a GPU-origin image into host RGBA/BGRA for this version.

The released `Mapika/decider-2b-vision` checkpoint at Hub revision
`863e290863655f1d6b69324d77d09ac972d21609` was converted to a 3.76 GB BF16
language GGUF plus a 671 MB BF16 Qwen3-VL projector. On the RX 9070, the
`media/cover.png` probe selected `food` with probability `0.9996`; its large
2000-by-840 input expanded to 1,638 vision tokens and completed in 0.97 s after
the Vulkan shader cache was populated.
Image dimensions therefore materially affect latency and should be bounded or
resized by the caller for real-time use.

An aligned 256-by-256 copy of the same image was also checked against a
Transformers 5.17 BF16 CPU reference with
`tools/generate_vision_parity_fixture.py`. Text tokens, answer slots, label
IDs, total text-plus-image token usage, structure, and categorical answer were
exact. The maximum deltas were `0.12444` for selected logits, `0.000033` for
probabilities, and `0.0003` across rounded response fields.

```powershell
python native\inferbridge_decider\tools\generate_vision_parity_fixture.py `
  C:\models\decider-2b-vision `
  native\inferbridge_decider\examples\simple_vision_choice.json `
  C:\images\probe-256.png out\parity\vision-python.json
python native\inferbridge_decider\tools\compare_parity.py `
  out\parity\vision-python.json out\parity\vision-native.json `
  --logit-atol 0.15 --probability-atol 0.001 --response-atol 0.001
```

For the original 2000-by-840 dimensions, Transformers chooses a 1984-by-832
grid while the pinned llama.cpp chooses 2016-by-832 at an exact half-step.
That is an upstream preprocessing-rounding difference, not a language-model
adapter difference. Use dimensions aligned to the projector's 32-pixel merge
factor for strict parity fixtures; production inputs remain valid at arbitrary
dimensions.

Generate the matching Python-side fixture from the original checkpoint with:

```powershell
python native\inferbridge_decider\tools\generate_parity_fixture.py `
  C:\models\decider-2b native\inferbridge_decider\examples\request.json `
  out\parity\python.json --device cuda --dtype bf16
```

Compare each row's `token_ids`, `slots`, and `selected_logits` with the native
diagnostic response before comparing the calibrated probabilities and final
response. Token and slot arrays must match exactly; logits and probabilities
use measured tolerances because the PyTorch and llama.cpp kernels differ.
The comparator defaults reflect the first BF16 CPU parity measurement
(`0.1` raw logits and `0.02` probabilities/rounded response fields); categorical
answers, token metadata, object structure, and token usage remain exact.

```powershell
python native\inferbridge_decider\tools\compare_parity.py `
  out\parity\python.json out\parity\native.json
```

`examples/request_packed_wide.json` is the packed conformance probe: it places
a 12-option wide Choice, Score, and Noul in one physical prompt so every slot
and per-type temperature is checked in a single chunked logical forward.

## BF16 parity status

The released `Mapika/decider-2b` v11 checkpoint was converted with the pinned
llama.cpp revision and checked on CPU against Transformers 5.17/PyTorch 2.11:

- Independent Choice plus isolated Score: exact tokens, slots, labels, usage,
  and categorical answer; maximum deltas were `0.05734` logits, `0.01152`
  probabilities, and `0.0137` across rounded numeric response fields.
- Packed 12-option Choice plus Score and Noul: exact tokens, slots, labels,
  usage, and categorical answers; maximum deltas were `0.07888` logits,
  `0.00417` probabilities, and `0.01` across numeric response fields.

These are compatibility measurements, not a claim of bitwise equality between
the PyTorch and llama.cpp kernels.

## Vulkan status

The Vulkan build was exercised with the BF16 v11 GGUF on an AMD Radeon RX 9070,
an AMD Radeon RX 6700 XT, and an NVIDIA GeForce GTX 1080. All ABI/protocol tests
passed. Each GPU was isolated with `GGML_VK_VISIBLE_DEVICES`; without that
variable llama.cpp exposes every supported discrete GPU and may split model
layers among them.

After each driver's shader cache was populated, three fresh-process runs per
GPU and example produced these median inference times:

| Example | Result | RX 9070 | RX 6700 XT | GTX 1080 | CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| `simple_choice.json` | `billing` (`p=0.9703`) | `54.76 ms` | `106.93 ms` | `146.23 ms` | `605.16 ms` |
| `simple_noul.json` | `0.7912` | `45.46 ms` | `100.40 ms` | `149.79 ms` | `593.73 ms` |
| `simple_score.json` | `1.32` | `55.67 ms` | `112.57 ms` | `124.83 ms` | `608.35 ms` |

The packed conformance probe passed against the Python BF16 fixture on every
GPU. Maximum `(logit, probability, rounded-response)` deltas were
`(0.07722, 0.00413, 0.0042)` on the RX 9070,
`(0.08388, 0.00501, 0.005)` on the RX 6700 XT, and
`(0.08272, 0.00346, 0.01)` on the GTX 1080.

Fresh canary model loads were about `2.28`–`2.31 s` on the RX 9070,
`2.40`–`2.43 s` on the RX 6700 XT, and `3.09`–`3.12 s` on the GTX 1080.
InferBridge keeps the loaded runtime alive, so this is startup cost rather than
per-request latency. The first inference for a new GPU/kernel shape can be much
slower while the driver compiles and caches Vulkan pipelines.

Run a timed Vulkan decision with:

```powershell
out\decider-native-vulkan\Release\decider_native_canary.exe `
  C:\models\decider-2b-native\decider-2b-bf16.gguf `
  native\inferbridge_decider\examples\simple_choice.json `
  VULKAN --timing
```

To pin an InferBridge process to one physical Vulkan device, set the zero-based
device index before process startup. Obtain the indices from
`vulkaninfo --summary`:

```powershell
$env:GGML_VK_VISIBLE_DEVICES = '0'
```

## Decider 4B v2.1 status

The harness is not tied to the 2B dimensions. The current
[`Mapika/decider-4b`](https://huggingface.co/Mapika/decider-4b) v2.1 checkpoint
(Hub commit `eb5fbdfc9448473ec25e399882912863afbdb70e`) was exported with the
same command and loaded without source changes. It is a 4.2B-parameter model;
the BF16 safetensors and GGUF payloads are 8.41 GB decimal (7.83 GiB).

The packed conformance probe passed against a Transformers BF16 CPU fixture on
both GPUs large enough for full offload. Maximum
`(logit, probability, rounded-response)` deltas were
`(0.05660, 0.00328, 0.0033)` on the RX 9070 and
`(0.03873, 0.00319, 0.0032)` on the RX 6700 XT. Tokens, slots, label IDs,
usage, object structure, and categorical answers were exact.

After shader-cache warm-up, three fresh-process runs produced these median
inference times:

| Example | RX 9070 | RX 6700 XT |
| --- | ---: | ---: |
| `simple_choice.json` | `93.20 ms` | `177.76 ms` |
| `simple_noul.json` | `92.54 ms` | `170.50 ms` |
| `simple_score.json` | `87.88 ms` | `178.39 ms` |

Fresh-process model loading was approximately `4.8 s`; a persistent InferBridge
runtime pays it once. Full BF16 offload does not fit the 8 GB GTX 1080 and
correctly fails with a Vulkan out-of-device-memory error. Leaving all Vulkan
GPUs visible allows llama.cpp to split the 4B model across them successfully.
A quantized 4B artifact would fit the GTX 1080, but it must pass separate
quality and calibration evaluation before being published as interchangeable
with the BF16 model.

The checkpoint's `decider_config.json` must remain beside the GGUF file. Use
`VULKAN` as the final argument only in a build configured with
`-DDECIDER_LLAMA_VULKAN=ON`; the harness now fails early when no Vulkan GPU
backend is actually registered.
