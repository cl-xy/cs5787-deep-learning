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

The notebook is a thin driver: it defines **no functions**. It sets up the
environment, imports the shared pipeline from [team_pipeline/](team_pipeline/) and the
method code from [xsa_method/](xsa_method/), holds the experiment settings, and calls
them. Run it from the repository root locally. In Colab, setup can clone the project:
set `PROJECT_REF` to a branch or commit containing `team_pipeline/runs.py`, or upload
the updated project. Unpublished local changes must be uploaded or pushed before Colab
can use them. The official DiT source, pretrained checkpoint, VAE and real-image FID
reference also require internet downloads.

**Where code lives:** common pipeline code (protocol, sampling, evaluation, metrics,
reports, run preparation) is in `team_pipeline/`; code specific to XSA or to the
method runner (attention filtering, block schedules, checkpoint restoration, frozen
block selection, method routing) is in `xsa_method/`; window looping is in
`loop_compression/`. New helper functions go in one of these packages, not in a
notebook cell. See [xsa_method/README.md](xsa_method/README.md) and the module table
below.

**Run fingerprint.** `prepare` stores a SHA-256 of the pipeline source in
`config.json` (and the files themselves in `code_sources.json`); every later step
refuses to run if the code differs. The sources are copied **when the packages are
imported**, so editing a file without restarting the kernel does not change the
fingerprint of the code that is actually running. After changing package code,
restart the kernel, rerun sections 1–4 and prepare a **new run**. Runs prepared
before this refactor (notebook-cell fingerprint, `notebook_sources.json`) cannot be
resumed; section 12 can still plot their saved artifacts. Plotting, the CLI and
the optional FID/CLIP code are not fingerprinted, so editing them does not
invalidate a prepared run.

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
  a held-out test split (30 per class, 300 per method).
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
| 2 Imports | package imports and protocol constants (no function definitions) | No |
| 3 Run tests | `tests_team/` and `tests_xsa/`, using tiny CPU models | No |
| 4 Configure | device (CUDA, then Apple MPS, then CPU), pinned DiT source, run folder | No |
| 5 Prepare | checkpoint download and hash check; records protocol, code and environment | No |
| 6 Select blocks | fixed `[4, 7]`, or calibration profiling | Optional |
| 7A–7E Validation | one cell per method, 20 images each | Yes |
| 8 Evaluation | paired LPIPS, both CLIP cosines, optional Clean-FID/reference preparation, comparison grid | No |
| 9 Custom method | optional member-defined method | Yes |
| 10 Held-out test | 300 images per method; off until `RUN_HELD_OUT_TEST = True` | Yes |
| 11 Export | zip the run folder | No |
| 12 Report | plots and tables from a completed run, without rerunning anything | No |

- **Where outputs go:** each run is written to `team_runs/<member>_<timestamp>/`. Inside
  it, `validation/` and `test/` hold manifests, per-image measurements, FP32 tensors,
  PNGs, `results.csv` and `results_extended.csv`.
- **Resuming:** set `RESUME_RUN` in section 4 to reopen a run. Completed method folders
  are never overwritten.
- **Choosing settings:** use only the validation split. The held-out test takes about 3.2
  hours of sampling for all five methods on Apple MPS.

## Other tools in the repository

The repository also contains a scripted version of the pipeline and a second study.
They share the notebook's code but default to an earlier protocol (8 ImageNet classes; a 1,000-image `fid` split; fixed damping
`alpha = 0.5`). Where they differ from the notebook, the notebook is the reference for
the main experiments.

| Path | Contents |
| --- | --- |
| [xsa_method/](xsa_method/) | XSA method package used by the notebook and the CLI: attention, schedules, checkpoints, selection and `bind_runner` method routing |
| [team_pipeline/](team_pipeline/) | shared pipeline used by the notebook and the command line: protocol and Imagenette classes (`data.py`), device/determinism helpers (`runtime.py`), profiling (`profiling.py`), sampling and `run_variant` (`sampling.py`), LPIPS evaluation (`evaluation.py`), Clean-FID/TPIPS and the Imagenette reference (`extended_metrics.py`, `imagenette.py`), report tables and figures (`report.py`), run preparation (`runs.py`), run fingerprint (`provenance.py`), notebook display helpers (`notebook.py`), FID/CLIP (`metrics.py`), CLI (`cli.py`). `experiment.py` only re-exports these for older imports |
| [loop_compression/](loop_compression/) | window looping (Chen et al., arXiv 2605.23872): loop a kept subset of a contiguous block window K times with naive, damped or anchored updates; sensitivity screening, validation driver, final benchmark |
| [DiT_Layer_Looping_GCP.ipynb](DiT_Layer_Looping_GCP.ipynb) | Colab/GCP entry point for the command-line pipeline, window-looping smoke tests and the original pilot; imports `team_pipeline` |
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
DIT_TEST_REPO="$PWD/third_party/DiT" python -m pytest --import-mode=importlib -q tests_xsa tests_team tests_loop
```

The notebook runs `tests_team` and `tests_xsa` in section 3. Use `--import-mode=importlib`:
both folders contain a `test_notebook.py`.

## No-loop block ablation studies

Use a **new prepared run** after updating this code (restart notebook kernels).
Install `transformers>=4.50,<5` before preparation; it is included in notebook
setup and both requirements files. The runner checks for it before inference.
The optional notebook section 13 and the module CLI use the same runner:

```bash
python -m team_pipeline.ablation sweep --output "$RUN"
python -m team_pipeline.ablation rules --output "$RUN"
python -m team_pipeline.ablation report --output "$RUN"
```

`sweep` profiles intact calibration trajectories and removes each of the 28 blocks
independently on validation pairs. `rules` fixes R=1: A removes minimum normalized
feature change; B has two variants: `B_before_max_delta` removes the immediate
predecessor and `B_after_max_delta` removes the immediate successor of the
largest feature change among blocks with that neighbor. Thus block 0 cannot
anchor the before variant, and the final block cannot anchor the after variant.
C removes the lower center of each depth third
(blocks 4, 14, 23 for 28 blocks). Ties use the smaller block index. All runs disable
layer reuse and use the intact DiT with identical class/seed/noise pairs as reference.
Each study evaluates paired LPIPS and CLIP image–text and image–image cosine scores.

Artifacts live under `ablations/{sweep,rules}/validation/`, separate from the main
method comparison, with dedicated calibration under `ablations/profile/`. Plans
are immutable. Rerunning skips complete, matching variants; incomplete variants
must be moved aside before retrying. `rules --split test` uses the same calibration
selection on held-out pairs. Full inference and first-use metric downloads can be
expensive; none of these studies runs by default.

The sweep exports a block-aligned CSV and PNG/PDF versions of delta versus LPIPS,
block position versus LPIPS, and a combined two-row heatmap. Depth thirds are
10/9/9 for 28 blocks (boundaries at 9.5 and 18.5); the heatmap uses separate color
scales and annotates raw values. `report` redraws figures from saved artifacts.

After the validation sweep, run cumulative top-R removal (no looping):

```bash
python -m team_pipeline.ablation topk --output "$RUN" --counts 1 2 3 4 6 8 12 16 20 24 28
```

Omit `--counts` to evaluate every R from 1 through 28. This deliberately removes
**highest** single-block LPIPS first (the most disruptive blocks), as specified;
ranking is frozen once from validation and ties favor smaller block indices.
Each R removes a prefix of that same ranking; it does not rerank combinations.
`--split test` reuses the validation ranking on held-out pairs.

`ablations/topk/` records the ranking and per-split frozen plans. Each split's
`analysis/` contains LPIPS and both CLIP curves including the intact R=0 baseline,
a metrics CSV with adjacent LPIPS increase per additional removed block, and
`largest_lpips_rise.json`. The largest positive slope is an exploratory candidate
interval for a sharp rise, not a statistically confirmed threshold; inspect the
curve and refine the tested counts around that interval in a new run. All three
metrics also retain the existing evaluator's per-image scores and sample SDs.

Ablation studies save images and metrics without retaining model checkpoints by default.
Notebook section 13 honors `KEEP_CHECKPOINTS` from section 4 (disabled by default).
Use `--keep-checkpoints` on the CLI, or set the notebook flag to `True`, to
retain a `model.pt` in each variant folder. Checkpoints consume several GB per variant; use a
suitably sized output drive. Restart the kernel and prepare a fresh run after
updating the pipeline. Retention is frozen in each study plan, and resuming with
retention enabled checks that saved checkpoints are present and complete in size.
