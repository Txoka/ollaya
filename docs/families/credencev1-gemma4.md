# Credence v1 Gemma4 E4B

Library name: `credencev1-gemma4:e4b`. This is a MiCA refinement of
EldanRing's Winnow-E4B, based on Google DeepMind's Gemma 4 E4B. It uses the existing
`winnow-v1` layout and llama.cpp runner; no new inference engine is necessary.
Adapter modifications are Apache-2.0. Preserve upstream LICENSE/NOTICE and the
training data's individual attribution requirements.

## Selected artifact

The selected checkpoint is `synth05-hardreal2-clip-25-s002500-T1calibrated`:
MiCA rank32/alpha32, clip25, 5% procedural synthetic questions, 19 real sources,
2,500 optimizer steps. Training used bitsandbytes NF4; the release candidate is
Q8_0 GGUF. Checkpoint-bound readout scaling is folded into weights and softcap;
external temperature is **1**, not Winnow's original 1.2574172. No new temperature
was fitted on benchmark data. This is supervised candidate-distribution training,
not a policy-gradient RL result.

The exact byte size and SHA256 are recorded in
`convert/releases/credencev1-gemma4-e4b.json`. Original adapter/source identities
remain recorded alongside the local export. This is the highest public-accuracy
candidate among the four Q8 exports tested; it is not a Pareto improvement over
original Winnow on all probability metrics.

## Benchmarks through Ollaya

Both columns use Q8_0, the same local Ollaya runtime and reserved benchmark rows.
Original Winnow uses its published temperature; Credence uses folded calibration
with external T1. Independent wire rounding limits reconstructed log metrics.

| Metric | Original Winnow-E4B | Credence v1 |
|---|---:|---:|
| Public macro accuracy, 13 subsets / 3,880 rows | 73.4567% | 74.1244% |
| Public pooled ECE, 10 bins | 7.0893% | 5.4789% |
| Public mean Brier | 0.380216 | 0.372403 |
| Typed accuracy, 400 states / 2,000 questions | 72.30% | 72.35% |
| Typed soft CE | 1.0623 | 1.0655 |
| Typed NLL | 0.6860 | 0.6872 |
| Typed ECE | 2.51% | 4.83% |

Public accuracy and calibration improve; typed accuracy differs by one question,
and typed probability metrics worsen. These are one-checkpoint measurements,
not significance claims. Retain per-subset numbers in
`results/credencev1-gemma4/`; do not substitute public pooled ECE for macro ECE.
Local latency is recorded in the artifacts, but should not be mixed with Ollaya's
RTX 4090/5090 leaderboard latency. No independent latency advantage is claimed.

## Packaging and verification

This addition follows merged model-addition PR
[49](https://github.com/ollaya-dev/ollaya/pull/49): family documentation belongs
here, with source pins, benchmark artifacts and generated registry manifests.
The published weights, LICENSE/NOTICE, data attribution, decision layout and
calibration metadata are available in the linked Hugging Face release below.

Both tags passed a clean pull into an initially empty store with no Hugging Face
token. The small registry manifests were served from an isolated localhost copy
of this branch's generated registry; weights came directly from the pinned public
Hugging Face URLs. This tests the proposed package before upstream deployment.
The client verified the actual SHA256 digests before writing either manifest.
Both packages pass CPU and CUDA parity against the pinned b11146 Python
reference: all 505 questions per device, unchanged edge cases plus 40 typed
states, truncation and twin-question cases, and the original 1e-3 log-softmax
tolerance. Token IDs, shared-prefix split points, candidates, rejections and
calibrated selections are checked. Exact reference metadata, fixture hashes and
probability differences are recorded in
`results/credencev1-gemma4/package-verification.json`.

No image support or Metal/Windows parity is claimed. No PR has been opened.

## Public-calibration variant

Alternative tag: `credencev1-gemma4:e4b-calibrated`. This is a different checkpoint,
trained with **0% synthetic**, not simply a temperature alias of the default.
Its residual temperature (approximately 1.040857) was fitted on the fixed 1,024
training-validation questions, not on either benchmark. Packaging must preserve
the full-precision fitted value and its validation provenance.

| Metric | Accuracy-focused tag | Public-calibration tag |
|---|---:|---:|
| Public macro accuracy | 74.1244% | 73.7228% |
| Public pooled ECE | 5.4789% | 3.7426% |
| Public Brier | 0.372403 | 0.369314 |
| Typed accuracy | 72.35% | 72.20% |
| Typed soft CE | 1.0655 | 1.0480 |
| Typed NLL | 0.6872 | 0.7011 |
| Typed ECE | 4.83% | 4.63% |

“Calibrated” describes the validation-fitted release and stronger public
calibration; it does not assert uniformly better calibration, perfect
probabilities or dominance over original Winnow. Its typed ECE remains above
original Winnow's 2.51%. Both tags retain the same source publication and package
verification procedure. Exact summary/provenance artifacts use the `calibrated-`
prefix in the results directory.

## Published source and package reproduction

The release is now public at
[Txoka/Credence-v1-Gemma4-E4B](https://huggingface.co/Txoka/Credence-v1-Gemma4-E4B),
pinned for this package to `7f7b4fa55dbb40ff47e3a30eb19e1e8dd77cedd2`.
`accuracy/model-Q8_0.gguf` and `calibrated/model-Q8_0.gguf` are distinct files,
each 8,718,469,216 bytes, with hashes in the release JSON. The catalog and generated
manifests now reference those actual files. Both clean public pulls are verified; independent package parity is recorded
separately from the earlier local-file quality benchmarks.

Recreate the small package inputs and registry without downloading the weights:

```shell
python convert/releases/prepare_credence.py
PYTHONPATH=convert python -m ollaya_convert.package credencev1-gemma4
```

The helper checks the published decision layout, expected GGUF hash and full
calibration values, then rebinds source metadata to the published byte-identical
weights. Packaging verifies HF's actual LFS digest/size against that pin. The
site build on Node22.17 requires `NODE_OPTIONS=--experimental-strip-types`.

To reproduce parity, generate each device's fixtures with
`python -m ollaya_convert.families.llm_common.export_llama winnow` using the
pinned release file, `--td 40`, and `--device cpu` or `CUDA0`. Use
`--temperature 1` for the accuracy tag or `--temperature 1.0408574437121012`
for the calibrated tag. Then run the upstream checker against the packaged
`decision.json`, `calibration.json` and the cleanly pulled `model.gguf`:

```shell
OLLAYA_LIBRARY_PATH=/path/to/lib/ollaya cargo run --release -p ollaya-runner --example parity_llama -- /path/to/package /path/to/goldens-cpu.jsonl cpu
OLLAYA_LIBRARY_PATH=/path/to/lib/ollaya cargo run --release -p ollaya-runner --example parity_llama -- /path/to/package /path/to/goldens-cuda.jsonl cuda
```

The website typecheck, 180-page build and link check pass. Local preview uses
`OLLAYA_ALLOW_MISSING_BLOBS=1` because 153 unrelated existing registry blobs are
absent from this checkout; both new packages' derived blobs are present.

## Image input

The separate vision packages reuse the exact unchanged F16 projector shipped
with Winnow E4B. Winnow's model card states that vision and audio modules stayed
frozen while its language tensors were trained; Credence's MiCA runs were
text-only. No multimodal training is claimed.

| Tag | Language checkpoint | Projector |
|---|---|---|
| `credencev1-gemma4:e4b-vision` | Same as `e4b` | Frozen Gemma4 E4B F16 |
| `credencev1-gemma4:e4b-calibrated-vision` | Same as `e4b-calibrated` | Same projector |

Projector: `vision/mmproj-Gemma4-E4B-F16.gguf` in the Credence HF repository,
990,372,672 bytes, SHA256
`ddf46c21d7078e95338cfc22306b19b276a29a5ad089023449dd54d4b6170a51`.
Its bytes match the pinned Winnow artifact. Source/base revisions and publication
pin are recorded in `convert/releases/credencev1-gemma4-e4b.json`.

The vision tags add one projector layer to the text packages; language weights
and decision/calibration configurations are shared. Text tags remain available
without downloading a projector. Image requests use the existing Winnow runtime
path described in [Winnow image documentation](winnow.md).

Both variants match stock llama-server b11146 on CPU, each on 21 image
requests / 65 decisions. Maximum conditional log-probability errors are
7.43e-6 (accuracy) and 7.62e-6 (calibrated), under the unchanged 1e-3 gate.
These comparisons use external temperature 1 to isolate runtime parity.
The calibrated package retains its external text-validation temperature.
The generated-color, multi-image ordering, repeat-request and escaped-state
fixtures test runtime parity, not broad visual reasoning accuracy. Text-only
calibration temperatures are not validated as image-specific calibration.

Image parity proof is recorded in
`convert/releases/credencev1-gemma4-vision-verification.json`. To reproduce,
start stock llama-server b11146 with the selected language GGUF and the pinned
projector (`--mmproj`), then export and check fixtures using the existing tool:

```shell
python -m ollaya_convert.families.winnow.vision_parity export http://127.0.0.1:REFERENCE_PORT /path/to/decision.json /tmp/vision.json
python -m ollaya_convert.families.winnow.vision_parity check http://127.0.0.1:RUNNER_PORT /tmp/vision.json
```

Run the Ollaya runner with the same language weights, projector and decision
configuration. Use CPU for both sides to reproduce the recorded result.

Both vision tags also passed unauthenticated pulls into an empty store. The
projector and both language files came from the pinned public HF revision and
passed digest verification. For this pre-publication check, only unpublished
Ollaya metadata-blob URLs were redirected to a local copy of this branch's
registry; language/projector URLs were unchanged.
