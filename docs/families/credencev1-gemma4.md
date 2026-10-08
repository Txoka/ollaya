# Credence v1 Gemma4 E4B

Proposed library name: `credencev1-gemma4:e4b`. This is a MiCA refinement of
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

## Release gates

This branch follows merged model-addition PR
[49](https://github.com/ollaya-dev/ollaya/pull/49), placing family documentation
here and preserving provenance rather than adding placeholder registry manifests.

Before a pullable library entry is added:

1. Publish the exact Q8 file, LICENSE, NOTICE, data attribution, model card,
   decision.json and calibration.json in the approved Hugging Face repository.
2. Pin its actual commit and file SHA in the catalog. Do not invent a URL or reuse
   Winnow's weights digest: these are different weights.
3. Run the existing GGUF exporter/package verification against that pinned source,
   preserving `thought: false`, the label token mapping and external T1.
4. Add site library metadata/content and README model row, then regenerate the
   registry/site catalog using the repository's normal tooling.
5. Verify the published pull and all primitive decisions on CPU and CUDA serially
   after the currently active serving tests. Existing local Ollaya benchmark
   results do not prove a future public package pulls correctly.

No image support or Metal/Windows parity is claimed. No PR has been opened.
