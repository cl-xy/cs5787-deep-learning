# Shared experiment workflow

All three members use `team_pipeline/data.py` and `team_pipeline/experiment.py`.
Only the method builder differs. The Mac and GCP notebooks import the same code
through their existing module names and also embed it for standalone upload.
For team comparisons, use the runner below: it adds protocol and environment checks.

## What stays fixed

| Component | Shared rule |
| --- | --- |
| Input | `team_protocol.json`: ImageNet labels and disjoint calibration/validation/test seeds |
| Default pilot | 16 calibration, 16 validation, 64 test images across 8 classes; 50 sampling steps |
| Preprocessing | CPU FP32 initial noise; 256x256 RGB decoded tensors in [-1,1]; no extra crops, resizing or per-image normalization |
| Sampling | Same official ancestral sampler, CFG=4, FP32, VAE revision, checkpoint hash and source commit |
| Method | Each member changes only a model builder or its JSON settings |
| Evaluation | Paired LPIPS AlexNet v0.1 on saved FP32 tensors; lower means closer to baseline, not better generation quality |
| Resources | Unique backbone parameters, actual checkpoint bytes and synchronized latency; CUDA peaks or MPS post-sample memory, labelled separately |
| Outputs | Same manifests, per-image CSVs, tensor images, PNG previews, summary CSV and comparison grid |

This is a training-free experiment. No training dataset is required. Calibration
selects candidate blocks; validation chooses method settings; test evaluates a
frozen choice. If a member later adds training, it belongs inside their builder;
sampling and evaluation still use the shared runner. Do not train on test inputs.
The default protocol is a pilot, not sufficient for final FID claims.

## Setup and run

Run commands from the repository root in the Python environment used by the
notebooks. On Mac use `requirements-mac.txt`; on a CUDA machine use
`requirements-experiment.txt` with the machine's CUDA-compatible PyTorch.
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
On GCP, pass `--repo vendor/DiT` to prepare/profile/run if using that checkout.

For different damping settings, use different names (e.g. `member3_alpha025`).
Choose settings using validation, then rerun baseline and the chosen methods with
`--split test` and evaluate that split. The runner prevents seed overlap and
changed input settings; the team must enforce the rule against tuning on test.
Changing the protocol, environment or device requires a new output root.
Compare timing on the same hardware; do not pool Mac and CUDA measurements.

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

## Adding distribution metrics later

FID, TPIPS, class-fidelity scoring and FLOP measurement are not implemented here.
Keep those additions in the shared evaluator when added, not in individual
member code. FID requires a larger, class-balanced evaluation and agreed ImageNet
reference statistics and preprocessing; LPIPS does not substitute for it.

## Tests

```bash
DIT_TEST_REPO="$PWD/third_party/DiT" python -m pytest --import-mode=importlib -q tests tests_local tests_team
```

Tests use tiny models and synthetic images without pretrained downloads. MPS
checks run only when the process can access MPS. CUDA execution requires a GPU
machine. A passing test suite verifies implementation contracts, not image quality.
