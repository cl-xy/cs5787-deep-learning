# Training-free compression of DiT-XL/2 by layer looping

CS5787 Deep Learning final project. We take the pretrained class-conditional
[DiT-XL/2](https://github.com/facebookresearch/DiT) (256×256 ImageNet, 28 transformer
blocks) and remove some of its blocks **without any retraining**. Each method then
decides how to fill the depth slots the removed blocks used to occupy. The question is
how close each method stays to the original model's images, and what it saves in
parameters, checkpoint size, FLOPs and latency.

## Methods

All methods drop the same blocks, so they have the same number of unique parameters.
They differ only in what runs at a removed slot.

| Method | CLI name | At each removed slot |
| --- | --- | --- |
| Original | `original` | nothing is removed (baseline) |
| Removal | `removal` | skip the block |
| Looping | `loop` | rerun the nearest preceding retained block |
| Damped looping | `damped` | `h + alpha * (B(h, c) - h)` with that block (default `alpha = 0.5`) |
| XSA reuse | `xsa_reuse` | rerun that block with exclusive self-attention: each head's output has its own value direction projected out |
| Window looping | `loop_compression.method:build_model` | loop a kept subset of a contiguous middle window K times (naive, damped or anchored), after Chen et al. (arXiv 2605.23872) |

Reused blocks are stored once, so looping reduces checkpoint size like removal does,
but it runs as many block evaluations as the original. Blocks are chosen either by
profiling feature change on calibration images or by a fixed list (default `[4, 7]`,
zero based).

## Experimental protocol

- [team_protocol.json](team_protocol.json) fixes the inputs: ImageNet class labels and
  disjoint seeds for the calibration (16), validation (16) and test (64) splits, plus a
  1,000-image `fid` split with one image per class.
- Sampling is fixed for every method: 50 ancestral steps, classifier-free guidance
  scale 4, FP32 with deterministic algorithms, the pinned DiT commit, checkpoint hash
  and VAE revision.
- Settings are chosen on validation only. Test and `fid` are run once, with frozen
  choices.
- Every run checks that the protocol, shared code and software environment are
  unchanged since `prepare`. It refuses to mix results that differ.

## Metrics

| Metric | What it measures |
| --- | --- |
| Paired LPIPS (AlexNet) | distance from the original model's image for the same label and seed; lower means closer, not better quality |
| Parameters, checkpoint size, DiT FLOPs | savings relative to the original |
| Latency, throughput, peak memory | synchronized end-to-end sampling time, on the same hardware only |
| FID, CLIP score | distribution quality and class agreement, on the `fid` split |
| Clean-FID and TPIPS (optional) | FID against real ImageNet images of the generated classes; aspect-wise perceptual distance from the original images |

With 500 or fewer images, FID is only meaningful relative to the original model, not
compared with published FID-50K numbers.

## Repository layout

| Path | Contents |
| --- | --- |
| [team_pipeline/](team_pipeline/) | shared runner: protocol and inputs (`data.py`), sampling and evaluation (`experiment.py`), block schedules (`methods.py`), XSA (`xsa.py`), FID/CLIP (`metrics.py`), Clean-FID/TPIPS and ImageNet references (`extended_metrics.py`), CLI (`cli.py`) |
| [loop_compression/](loop_compression/) | window-looping method: block wrapper, one-step sensitivity screening, validation driver, final benchmark |
| [DiT_Layer_Looping_GCP.ipynb](DiT_Layer_Looping_GCP.ipynb) | Google Colab / GCP entry point |
| [tests_team/](tests_team/), [tests_loop/](tests_loop/) | CPU tests with tiny models, no downloads |
| `third_party/DiT` | official DiT source (git submodule, pinned commit) |
| [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) | detailed rules, every command, metric definitions |

## Running on Google Colab (recommended)

1. Open [DiT_Layer_Looping_GCP.ipynb](DiT_Layer_Looping_GCP.ipynb) in Colab and pick a
   GPU runtime (**Runtime > Change runtime type**).
2. Run **Part 0**, cells 0.1 and 0.2. They clone the `milestone-check` branch with its
   submodule, optionally cache the 2.7 GB checkpoint in Google Drive, and install the
   dependencies. Level A runs the unit tests. Levels B and C are optional smoke tests of
   window looping.
3. Run **Part X** for the five-method comparison (original, removal, loop, damped,
   XSA reuse). Run its cells in order: prepare, select blocks, validation runs, LPIPS
   evaluation, then the optional Clean-FID/TPIPS step. The held-out test cell is off
   until you set `RUN_HELD_OUT_TEST = True`.
4. The last cell copies tables, manifests and preview images to Google Drive.

For FID against real ImageNet images, set `BUILD_IMAGENET_REFERENCE = True` in Part X.4.
You need a Hugging Face account with access to the gated `ILSVRC/imagenet-1k` dataset.

Sections 1-9 at the end of the notebook are an earlier standalone pilot and are not
needed.

## Running locally (CLI)

```bash
git clone --recurse-submodules -b milestone-check https://github.com/cl-xy/cs5787-deep-learning.git
cd cs5787-deep-learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt   # on CUDA machines, keep your CUDA build of torch/torchvision
```

The device is chosen automatically (CUDA, then Apple MPS, then CPU). Pass
`--device cuda:0`, `mps` or `cpu` to `prepare` to choose explicitly. Run commands from
the repository root:

```bash
RUN=team_runs/pilot
python -m team_pipeline.cli prepare --output $RUN          # downloads and verifies the checkpoint
python -m team_pipeline.cli select  --output $RUN --removed 4 7
# or choose blocks by calibration: python -m team_pipeline.cli profile --output $RUN --remove-count 2

python -m team_pipeline.cli run --output $RUN --split validation --method original  --name original
python -m team_pipeline.cli run --output $RUN --split validation --method removal   --name member1_removal
python -m team_pipeline.cli run --output $RUN --split validation --method loop      --name member2_loop
python -m team_pipeline.cli run --output $RUN --split validation --method damped    --name member3_damped --alpha 0.5
python -m team_pipeline.cli run --output $RUN --split validation --method xsa_reuse --name member4_xsa_reuse

python -m team_pipeline.cli evaluate --output $RUN --split validation \
    --names original member1_removal member2_loop member3_damped member4_xsa_reuse
```

Each run writes to `$RUN/<split>/<name>/`: a manifest, `samples.csv` with per-image
latency and memory, FP32 image tensors and PNGs. `evaluate` writes `results.csv`,
`paired_metrics.csv` and `comparison_grid.png` to the split folder. Existing run
folders are never overwritten. Use `--limit N` for quick smoke runs.

Optional and final-study commands:

```bash
# Real-image FID reference for the generated classes, then Clean-FID / TPIPS
huggingface-cli login
python -m team_pipeline.cli fid-reference --output $RUN --split validation --images-per-class 50
python -m team_pipeline.cli extended --output $RUN --split validation --names original member4_xsa_reuse \
    --fid-reference-dir fid_references/<folder printed above> --tpips-factors overall shape texture

# Window looping: sensitivity screening -> validation -> frozen final benchmark
python -m loop_compression.sensitivity blocks  --output $RUN
python -m loop_compression.sensitivity windows --output $RUN
python -m loop_compression.drive validate --output $RUN
python -m loop_compression.drive freeze   --output $RUN
python -m loop_compression.final test   --output $RUN
python -m loop_compression.final fid    --output $RUN --budget 4 --stride 2 --limit 500
python -m loop_compression.final report --output $RUN
```

To add your own method, write a `build_model(model, settings)` function and run it
with `--method your.module:build_model --method-config settings.json`. See
[team_pipeline/example_method.py](team_pipeline/example_method.py) and
[TEAM_WORKFLOW.md](TEAM_WORKFLOW.md).

## Tests

```bash
DIT_TEST_REPO="$PWD/third_party/DiT" python -m pytest --import-mode=importlib -q tests_team tests_loop
```

The tests use tiny random models and synthetic images on CPU. They check the
implementation's contracts (schedules, XSA, resumability, metric bookkeeping), not
image quality.
