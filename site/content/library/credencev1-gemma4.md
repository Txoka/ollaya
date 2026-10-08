Credence v1 by [Txoka](https://huggingface.co/Txoka/Credence-v1-Gemma4-E4B)
refines EldanRing's Winnow-E4B using MiCA. Both variants run their Q8_0 GGUF files
through Ollaya's existing Winnow prompt and llama.cpp engine.

| Tag | Public accuracy | Public ECE ↓ | Typed accuracy | Typed ECE ↓ |
|---|---:|---:|---:|---:|
| Original `winnow:e4b` | 73.46% | 7.09% | 72.30% | 2.51% |
| `credencev1-gemma4:e4b` | 74.12% | 5.48% | 72.35% | 4.83% |
| `credencev1-gemma4:e4b-calibrated` | 73.72% | 3.74% | 72.20% | 4.63% |

Local measurements through Ollaya: 13 public subsets/3,880 rows and 400 typed
states/2,000 questions. Public ECE is pooled ten-bin ECE. These are single-run
results, not significance claims. Neither variant improves every metric over
original Winnow. Full probability metrics and per-subset results are retained in
the repository's `results/credencev1-gemma4/` directory. No comparable RTX 4090
latency has been measured for these releases.

```shell
ollaya run credencev1-gemma4:e4b --preset triage "My order never arrived. Please refund it."
ollaya run credencev1-gemma4:e4b-calibrated --preset triage "My order never arrived. Please refund it."
```

The accuracy tag was trained with 5% procedural synthetic questions; the calibrated
tag is a separate 0%-synthetic checkpoint. Both start from calibrated Winnow.
Their readout scaling is folded into the weights and softcap. The first uses
external temperature 1; the second uses validation-fitted 1.0408574437121012.
No temperature was fitted on benchmark data.

Weights are 8.72 GB each, downloaded from a commit-pinned public Hugging Face
repository and verified by SHA256. Text only; up to 64 candidate options with the
existing Winnow context limits. No image support is claimed. Apache-2.0; preserve
Winnow/Gemma4 notices and individual training-data attribution.

Both published files passed clean, unauthenticated downloads and SHA256 verification.
Both packages also pass the upstream CPU/CUDA parity suite. Exact evidence and
reproduction instructions are in the family documentation.
