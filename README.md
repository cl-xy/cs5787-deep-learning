# Training-free compression of DiT-XL/2 by layer looping

CS5787 Deep Learning final project. We take the pretrained class-conditional
[DiT-XL/2](https://github.com/facebookresearch/DiT) (256×256 ImageNet, 28 transformer
blocks) and remove some of its blocks **without any retraining**. Each method then
decides what runs in the depth slots the removed blocks used to occupy. The question is
how close each method stays to the original model's images and to real images, and what
it saves in parameters, checkpoint size and latency.

The main experiments are in
**[DiT_Layer_Looping_colab.ipynb](DiT_Layer_Looping_colab.ipynb)**. The interim report
built from them is in [report/](report/) ([PDF](report/interim_report.pdf)).

## Main experiments: `DiT_Layer_Looping_colab.ipynb`

The notebook is self-contained. It defines the whole pipeline in its own cells, so it
does not import this repository's Python packages. It needs internet access only for the
official DiT source, the pretrained checkpoint, the VAE and the real-image FID reference.
Upload it on its own to Colab, or run it from the repository root in a local Python
environment.

### Methods

Every compressed method removes the same blocks (default `[4, 7]`, zero based), so all
have the same number of unique parameters. They differ only in what runs at a removed
slot:

| Method | Run name | At each removed slot |
| --- | --- | --- |
| Original | `original` | nothing is removed (baseline) |
| Removal | `member1_removal` | skip the block |
| Looping | `member2_loop` | rerun the nearest preceding retained block |
| Damped looping | `member3_damped` | `h + (B(h, c) - h) / k`, where `k` counts calls of that block in the forward pass (its own call is `k = 1`, so the first reuse uses 1/2) |
| XSA reuse | `member4_xsa_reuse` | rerun that block with exclusive self-attention: each head's output has its own value direction projected out (training-free, reused calls only) |

Reused blocks are stored once, so looping shrinks the checkpoint as much as removal does
while keeping all 28 block evaluations. Blocks are frozen as a fixed list by default. Set
`CALIBRATE_BLOCKS = True` to rank blocks by their normalized feature change on the
calibration images instead.

### Protocol and metrics

- **Classes:** the ten Imagenette classes, identified by their ImageNet ids (0, 217,
  482, 491, 497, 566, 569, 571, 574, 701).
- **Splits:** disjoint seeds for calibration (20 images), validation (20 per method) and
  a held-out test split (500 per class, 5,000 per method).
- **Sampling:** identical for every method: 50 ancestral steps, classifier-free guidance
  scale 4, FP32 and deterministic kernels, with the DiT commit, checkpoint hash and VAE
  revision all pinned.
- **Paired LPIPS (AlexNet):** distance from the original model's image for the same
  label and seed. Lower means closer to the original, not better quality.
- **CLIP image–text cosine:** raw cosine between each generated image and
  `a photo of a {ImageNet class name}`, using CLIP ViT-B/32. Higher is better.
- **CLIP image–image cosine:** raw cosine against the original model image with
  the same class and seed. Higher means greater semantic preservation, not absolute
  image quality. Both CLIP metrics are enabled by default (`ENABLE_CLIP=True` in
  section 8), run in batches of 16 on CPU, and are independent of the FID toggle.
  First use downloads CLIP weights/tokenizer. Per-method means and sample standard
  deviations are saved in `results_extended.csv`, per-image scores in
  `paired_metrics_extended.csv`, and Section 12 includes them in tables and plots.
- **FID (optional, off by default):** Set `ENABLE_FID = True` in section 8 to enable
  reference preparation and evaluation for validation and held-out test. With it off,
  reference downloads, Clean-FID installation and FID computation are skipped. Clean-FID compares against 5,000 real images, 500 per class from the
  Imagenette v2
  training split. These are downloaded once through torchvision (about 1.5 GB) and
  center-cropped to 256. With 20 validation images, FID is exploratory.
- **TPIPS (optional, off by default):** aspect-wise distance (overall, shape, texture)
  from the original images.
- **Efficiency:** unique parameters, checkpoint size, block evaluations per forward pass
  and synchronized latency per image.

### Running it

Run the sections in order:

| Section | What it does | Generates images? |
| --- | --- | --- |
| 1 Setup | workspace, optional Google Drive storage, dependencies | No |
| 2 Definitions | pipeline, protocol, methods, XSA and tests as notebook cells | No |
| 3 Run tests | CPU contract tests with tiny models | No |
| 4 Configure | device (CUDA, then Apple MPS, then CPU), pinned DiT source, run folder | No |
| 5 Prepare | checkpoint download and hash check; records protocol, code and environment | No |
| 6 Select blocks | fixed `[4, 7]`, or calibration profiling | Optional |
| 7A–7E Validation | one cell per method, 20 images each | Yes |
| 8 Evaluation | paired LPIPS, both CLIP cosines, optional Clean-FID/reference preparation, comparison grid | No |
| 9 Custom method | optional member-defined method | Yes |
| 10 Held-out test | 5,000 images per method; off until `RUN_HELD_OUT_TEST = True` | Yes |
| 11 Export | zip the run folder | No |
| 12 Report | plots and tables from a completed run, without rerunning anything | No |

- **Where outputs go:** each run is written to `team_runs/<member>_<timestamp>/`. Inside
  it, `validation/` and `test/` hold manifests, per-image measurements, FP32 tensors,
  PNGs, `results.csv` and `results_extended.csv`.
- **Resuming:** set `RESUME_RUN` in section 4 to reopen a run. Completed method folders
  are never overwritten.
- **Choosing settings:** use only the validation split. The held-out test takes about 53
  hours of sampling for all five methods on Apple MPS.

## Other tools in the repository

The repository also contains a scripted version of the pipeline and a second study.
They use an earlier protocol (8 ImageNet classes; a 1,000-image `fid` split; fixed damping
`alpha = 0.5`). Where they differ from the notebook, the notebook is the reference for
the main experiments.

| Path | Contents |
| --- | --- |
| [team_pipeline/](team_pipeline/) | command-line runner: protocol (`data.py`), sampling and evaluation (`experiment.py`), block schedules (`methods.py`), XSA (`xsa.py`), FID/CLIP (`metrics.py`), Clean-FID/TPIPS (`extended_metrics.py`), CLI (`cli.py`) |
| [loop_compression/](loop_compression/) | window looping (Chen et al., arXiv 2605.23872): loop a kept subset of a contiguous block window K times with naive, damped or anchored updates; sensitivity screening, validation driver, final benchmark |
| [DiT_Layer_Looping_GCP.ipynb](DiT_Layer_Looping_GCP.ipynb) | Colab entry point for the command-line pipeline and window-looping smoke tests |
| [tests_team/](tests_team/), [tests_loop/](tests_loop/) | CPU tests with tiny models, no downloads |
| `third_party/DiT` | official DiT source (git submodule, pinned commit) |
| [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) | rules, every command and metric definitions for the command-line pipeline |

### Command-line pipeline

```bash
git clone --recurse-submodules -b milestone-check https://github.com/cl-xy/cs5787-deep-learning.git
cd cs5787-deep-learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt   # on CUDA machines, keep your CUDA build of torch/torchvision

RUN=team_runs/pilot
python -m team_pipeline.cli prepare --output $RUN
python -m team_pipeline.cli select  --output $RUN --removed 4 7
python -m team_pipeline.cli run --output $RUN --split validation --method original --name original
python -m team_pipeline.cli run --output $RUN --split validation --method xsa_reuse --name member4_xsa_reuse
python -m team_pipeline.cli evaluate --output $RUN --split validation --names original member4_xsa_reuse

# Window looping: sensitivity screening -> validation -> frozen final benchmark
python -m loop_compression.sensitivity windows --output $RUN
python -m loop_compression.drive validate --output $RUN
python -m loop_compression.drive freeze   --output $RUN
python -m loop_compression.final test     --output $RUN
```

See [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) for the full command set: removal, loop and
damped runs, FID-500 and CLIP score, timing, and custom methods.

### Tests

```bash
DIT_TEST_REPO="$PWD/third_party/DiT" python -m pytest --import-mode=importlib -q tests_team tests_loop
```

The notebook runs its own test suite in section 3.
