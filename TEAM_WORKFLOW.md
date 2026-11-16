# Shared experiment workflow

All three members use `team_pipeline/data.py` and `team_pipeline/experiment.py`.
Only the method builder differs. The Mac notebook imports project code from
`dit_looping` and embeds the required files for standalone use.
For team comparisons, use the runner below: it adds protocol and environment checks.

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

python -m team_pipeline.cli evaluate --output team_runs/pilot --split validation --names original member1_removal member2_loop member3_damped
```

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

FID-1K and CLIP score are computed from the `fid` split (below), not from the small
validation/test pilot sets. LPIPS measures paired drift from the baseline and does
not substitute for either absolute quality metric.

## FID-1K and CLIP score (`fid` split)

`team_protocol.json` has a `fid` split: one image for each of the 1,000 ImageNet
classes (labels 0-999, seeds 10096-11095, disjoint from every other split). It is
used only for final FID/CLIP, never to choose settings.

**Everyone must re-run `prepare` on a NEW output root.** Adding the split changed
`data.py` and `team_protocol.json`, so the shared-code and protocol hashes changed and
older roots are rejected on purpose. Results from old roots are not comparable.

Extra requirements (already in `requirements-mac.txt`): `transformers>=4.50,<5` for
CLIP and `torch-fidelity` for Inception features. Keep `huggingface-hub<1`, which
`diffusers` needs; `transformers` 5.x pulls in hub 2.x and breaks that pin.

```bash
# 1000 images per variant, about 2.7 h on a Mac at batch size 1. --limit N runs a smoke subset.
python -m team_pipeline.cli run --output team_runs/main --split fid --method original --name original
python -m team_pipeline.cli run --output team_runs/main --split fid --method removal --removed 12 15 --name rm_12_15
python -m team_pipeline.cli clip --output team_runs/main --split fid --names original rm_12_15
python -m team_pipeline.cli fid  --output team_runs/main --split fid --names original rm_12_15 \
    --reference checkpoints/reference/VIRTUAL_imagenet256_labeled.npz
```

- `clip`: CLIP ViT-B/32 cosine similarity with "a photo of a {class name}" (torchvision
  ImageNet names). `clip_score` is 100 x mean(max(cos, 0)).
- `fid`: TF-FID-compatible Inception features (torch-fidelity) of the saved tensors.
  Reports `fid_vs_original` (distance to the unmodified model's own 1,000 images) and,
  with `--reference`, `fid_vs_adm_imagenet256` against the ADM ImageNet-256 reference
  statistics (2 GB, `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz`).
- With 1,000 images the 2048-d covariance is rank deficient and absolute FID is biased
  high. Call it FID-1K and compare variants only, never against published FID-50K.
- Both commands write `fid.json` / `clip.json` per variant and refuse variants whose
  configuration, runtime or shared code differs from `original`.
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
