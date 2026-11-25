# Shared experiment workflow

All three members use the shared `team_pipeline/` package (`data.py` protocol,
`sampling.py`, `evaluation.py`, `runs.py`). Only the method builder differs. The
Colab notebook and the command line import the same code; neither embeds a copy,
and no helper package is generated at run time. For team comparisons, use the
runner below or the notebook: both add protocol and environment checks.

| Code that is... | Belongs in |
| --- | --- |
| common to every method (protocol, sampling, metrics, reports, run preparation) | `team_pipeline/` |
| specific to XSA or the method runner (attention, schedules, selection, routing) | `xsa_method/` |
| specific to window looping | `loop_compression/` |
| a member's own method | a new module copied from `team_pipeline/example_method.py` |

Do not define functions in notebook cells. Notebooks only import, configure and call.

**Run fingerprint:** `prepare` records a hash of the pipeline source (`team_pipeline`
sampling/evaluation modules plus `xsa_method`) taken at import time, and the files
themselves in `code_sources.json`. Every later step and every comparison requires the
same hash. Editing a file without restarting the kernel does not change it. After
changing pipeline code, restart and `prepare` a new output root.

## What stays fixed

| Component | Shared rule |
| --- | --- |
| Input | `team_protocol.json`: ImageNet labels and disjoint calibration/validation/test seeds, plus a 1,000-image `fid` split for final FID/CLIP |
| Default pilot | 16 calibration, 16 validation, 64 test images across 8 classes; 50 sampling steps |
| Preprocessing | CPU FP32 initial noise; 256x256 RGB decoded tensors in [-1,1]; no extra crops, resizing or per-image normalization |
| Sampling | Same official ancestral sampler, CFG=4, FP32, VAE revision, checkpoint hash and source commit |
| Method | Each member changes only a model builder or its JSON settings |
| Evaluation | Paired LPIPS AlexNet v0.1 on saved FP32 tensors; lower means closer to baseline, not better generation quality |
| Resources | Unique parameters, checkpoint bytes, estimated DiT FLOPs, synchronized latency, throughput and baseline-relative reductions/speedup; CUDA peaks or MPS post-sample memory |
| Outputs | Same manifests, per-image CSVs, tensor images, PNG previews, summary CSV and comparison grid |

This is a training-free experiment. No training dataset is required. Calibration
selects candidate blocks; validation chooses method settings; test evaluates a
frozen choice. If a member later adds training, it belongs inside their builder;
sampling and evaluation still use the shared runner. Do not train on test inputs.
The default protocol is a pilot, not sufficient for final FID claims.

## Setup and run

Run commands from the repository root in the Python environment used by the
notebooks. On Mac use `requirements-mac.txt`.
The official source must be at the commit pinned in the protocol.

```bash
python -m team_pipeline.cli prepare --output team_runs/pilot --device auto
python -m team_pipeline.cli profile --output team_runs/pilot --remove-count 2

python -m team_pipeline.cli run --output team_runs/pilot --split validation --method original --name original
python -m team_pipeline.cli run --output team_runs/pilot --split validation --method removal --name member1_removal
python -m team_pipeline.cli run --output team_runs/pilot --split validation --method loop --name member2_loop
python -m team_pipeline.cli run --output team_runs/pilot --split validation --method damped --name member3_damped --alpha 0.5
python -m team_pipeline.cli run --output team_runs/pilot --split validation --method xsa_reuse --name member4_xsa_reuse

python -m team_pipeline.cli evaluate --output team_runs/pilot --split validation --names original member1_removal member2_loop member3_damped member4_xsa_reuse
```

To skip calibration and freeze explicit blocks instead of `profile`, run
`python -m team_pipeline.cli select --output team_runs/pilot --removed 4 7` (zero-based).
A run's selection is frozen; a different selection needs a new output root.
`--alpha` applies only to `damped` (default 0.5); the other built-ins always use 1.0.

`prepare` downloads the checkpoint if missing and checks its hash. Generation also
downloads the pinned VAE if needed. The first real LPIPS evaluation may download
AlexNet weights. Preparation records the resolved device and package versions.
Use `--device cpu`, `--device mps` or `--device cuda:0` to select explicitly.

For different damping settings, use different names (e.g. `member3_alpha025`).
Choose settings using validation, then rerun baseline and the chosen methods with
`--split test` and evaluate that split. The runner prevents seed overlap and
changed input settings; the team must enforce the rule against tuning on test.
Changing the protocol, environment or device requires a new output root.
Compare timing on the same hardware and software environment.

## XSA reuse (`--method xsa_reuse`)

Like `loop`, each removed slot reruns the nearest preceding retained block, so the
unique parameters equal those of `removal`. At those replacement slots only, the
block's attention uses exclusive self-attention: each token's per-head attention
output has its own value direction projected out before the output projection
(`team_pipeline/xsa.py`). Ordinary block calls are unchanged; the rest of the
DiT block (adaLN, gates, residuals, MLP) runs as usual. The schedule metadata in
each manifest marks the XSA slots.

Optional extra metrics on a finished, evaluated split (`team_pipeline/extended_metrics.py`):

```bash
pip install datasets clean-fid tpips "scipy<1.18"   # only the ones you enable
huggingface-cli login   # account with access to the gated ILSVRC/imagenet-1k dataset
python -m team_pipeline.cli fid-reference --output team_runs/pilot --split validation --images-per-class 50
python -m team_pipeline.cli extended --output team_runs/pilot --split validation \
    --names original member1_removal member2_loop member3_damped member4_xsa_reuse \
    --fid-reference-dir fid_references/imagenet_validation_256_50pc_88_207_279_360_387_417_974_979 \
    --tpips-factors overall shape texture
```

`fid-reference` reads the classes from the split's `original` manifest and saves the
first N ImageNet validation images of each class (RGB, aspect-preserving bicubic
resize, 256x256 center crop, PNG) to `fid_references/`, which git ignores. It records
the settings and a description in `reference.json`, so `extended` needs no
`--fid-reference-description` for that folder, and a complete folder is reused.
It downloads the whole validation split (about 6 GB) unless you pass `--streaming`.
For any other folder of real images, pass `--fid-reference-description`.
The `extended` command computes both CLIP cosine metrics by default, independently
of FID and TPIPS. Use `--no-clip` to skip CLIP, `--clip-device cpu` to select its
device, and `--clip-batch-size 16` to control memory use. Install `transformers>=4.50,<5`
for CLIP. Image–text cosine uses `a photo of a {ImageNet class name}`; image–image
cosine compares with `original` at the same label/seed. Both are raw cosine in
[-1, 1], higher is better. Extended summary columns are
`clip_image_text_cosine_mean/std` and `clip_image_image_cosine_mean/std`
(with `_mean` and `_std` as separate suffixes). Per-image columns omit the suffix.
With `--no-clip` and no FID/TPIPS options, `extended` only copies LPIPS results.
Outputs go to `results_extended.csv`, `paired_metrics_extended.csv` and
`evaluation_extended.json`; the LPIPS files are not modified. Clean-FID here uses a
folder of real images; for the final FID-500 study use the `fid` command below.
The Colab entry point for this comparison is Part X of `DiT_Layer_Looping_GCP.ipynb`.

## Your own method

Copy `team_pipeline/example_method.py` to a new importable module and implement:

```python
def build_model(model, settings):
    # model is the shared pretrained DiT on CPU.
    # Modify its blocks/weights here, then return it.
    return model
```

The shared pipeline freezes the returned model and handles noise, conditioning,
sampling, decoding, saving, timing and evaluation. Keep the DiT forward interface,
FP32 weights and architecture compatible. A custom container should expose
`metadata()` describing its schedule and `retained` containing unique blocks;
otherwise block counts assume a plain, once-per-block sequence.

```bash
python -m team_pipeline.cli run --output team_runs/pilot --split validation --method team_pipeline.example_method:build_model --method-config team_pipeline/example_method.json --name custom_example
```

The example settings are illustrative, not selected or validated. The manifest
records custom settings and a hash of the builder's source file. Commit any
additional helper files with the method so it can be reproduced.
Built-in scheduled checkpoints can be restored with `load_compact_checkpoint`.
For another custom architecture, rebuild it with the method code before loading
its saved `state_dict`; the generic loader does not reconstruct arbitrary methods.

## Metric interpretation and distribution metrics

The summary reports parameter/checkpoint reduction, estimated DiT sampling FLOPs,
latency reduction, speedup and throughput relative to `original`. FLOPs count a
multiply-add as two operations and cover the CFG DiT forwards; elementwise ops,
normalization, softmax, sampler arithmetic and VAE decoding are excluded. Latency
remains the authoritative end-to-end compute measurement.

FID-N and CLIP score are computed from a deterministic subset of the `fid` split (below), not from the small
validation/test pilot sets. LPIPS measures paired drift from the baseline and does
not substitute for either absolute quality metric.

## FID-N and CLIP score (`fid` split)

`team_protocol.json` has a `fid` split: one image for each of the 1,000 ImageNet
classes (labels 0-999, seeds 10096-11095, disjoint from every other split). It is
used only for final FID/CLIP, never to choose settings. The final 500-image study
uses `--stride 2 --limit 500`, which applies stride before limit and therefore
selects protocol entries 0, 2, ..., 998: 500 classes spread across ImageNet.

**Everyone must re-run `prepare` on a NEW output root.** Adding the split changed
`data.py` and `team_protocol.json`, so the shared-code and protocol hashes changed and
older roots are rejected on purpose. Results from old roots are not comparable.

Extra requirements (already in `requirements-mac.txt`): `transformers>=4.50,<5` for
CLIP and `torch-fidelity` for Inception features. Keep `huggingface-hub<1`, which
`diffusers` needs; `transformers` 5.x pulls in hub 2.x and breaks that pin.

```bash
# Final FID-500/CLIP study: original, frozen budget-4 loop and equal-parameter removal.
# On a Mac, generation takes about 4.2 hours. Keep the machine awake.
caffeinate -i python -m loop_compression.final fid --output team_runs/main --budget 4 \
    --stride 2 --limit 500 \
    --reference checkpoints/reference/VIRTUAL_imagenet256_labeled.npz
python -m loop_compression.final report --output team_runs/main
```

For direct runner calls, `--stride N` selects every Nth protocol pair before
`--limit` truncates the selection. The exact selected pairs are stored in each
manifest. `loop_compression.final` skips an existing variant only if its manifest,
`samples.csv` rows and saved tensors form the complete expected set. A partial or
mismatched folder stops the command with its path. Move that folder aside manually
after inspecting it, then rerun; no output is deleted automatically.

- `clip`: CLIP ViT-B/32 cosine similarity with "a photo of a {class name}" (torchvision
  ImageNet names). `clip_score` is 100 x mean(max(cos, 0)).
- `fid`: TF-FID-compatible Inception features (torch-fidelity) of the saved tensors.
  Reports `fid_vs_original` (distance to the unmodified model's corresponding images)
  and, with `--reference`, `fid_vs_adm_imagenet256` against the ADM ImageNet-256
  reference statistics (2 GB, `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz`).
- The metric name records the actual count, such as `FID-500 (relative use only)`.
  At 500 images, the 2048-d covariance is rank deficient and absolute FID is biased
  upward. Use FID versus the original's corresponding 500 images as the main result.
  Treat FID versus ADM's 50K statistics as secondary, and never compare it with
  published FID-50K.
- Both commands write `fid.json` / `clip.json` per variant and refuse variants whose
  configuration, runtime, shared code or selected pairs differ from `original`.
  Their JSON includes the actual image count and distinct class count.
- `run --split fid` has no resume: a crash loses that variant. Existing folders are
  never overwritten.

## Looped-block compression method (`loop_compression/`)

Training-free looping of a middle window of DiT blocks (Chen et al., arXiv 2605.23872).
Run order, all on validation until configs are frozen:
`python -m loop_compression.sensitivity blocks|windows --output ROOT`, then
`python -m loop_compression.drive validate|freeze --output ROOT`, then
`python -m loop_compression.final test|timing|fid|report --output ROOT`.
Details are in the module docstrings; `final` refuses to run without `frozen_configs.json`.

## Tests

```bash
DIT_TEST_REPO="$PWD/third_party/DiT" python -m pytest --import-mode=importlib -q tests tests_team tests_loop
```

Tests use tiny models and synthetic images without pretrained downloads. MPS
checks run only when the process can access MPS. A passing test suite verifies
implementation contracts, not image quality.
