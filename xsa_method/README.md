# XSA method for DiT

This package holds the XSA method implementation used by
`DiT_Layer_Looping_colab.ipynb` and the `team_pipeline` CLI. The notebook owns only
experiment settings; shared sampling, evaluation and reporting are in
`team_pipeline/`, and this package owns attention modifications, block schedules and
method routing.

| Module | Responsibility |
| --- | --- |
| `attention.py` | Project each attention head's output off its current self-value direction, using hooks removed after each call, including on errors |
| `schedule.py` | Register retained blocks once, execute the schedule, apply XSA only at reused slots, and record original-index metadata |
| `checkpoint.py` | Rebuild compact schedules before loading state dictionaries |
| `selection.py` | Read/write a run's frozen, zero-based block selection (`depth` defaults to 28) |
| `experiment.py` | Validate method arguments (`run_method`) and bind them to a run and shared runner once (`bind_runner`) |
| `__init__.py` | Public imports and the package source snapshot used in run fingerprints |

## Use with a model

From the repository root, with PyTorch and timm installed:

```python
from xsa_method import ScheduledBlocks

# model is a pretrained official DiT, before installing a schedule.
model.blocks = ScheduledBlocks(
    model.blocks, removed=[4, 7], mode="xsa_reuse", alpha=1.0,
)
model.eval().requires_grad_(False)
```

Retained calls use the original attention. At removed slots 4 and 7, the schedule
reruns retained blocks 3 and 6 with XSA. Their parameters are registered once.
The DiT block's adaLN, gates, residuals and MLP remain on their original path.
The hook adapter targets standard timm `Attention` and sequential eager inference.

The same scheduler supports `original`, `removal`, `loop`, and `damped` comparisons.
Damped mode defaults to `alpha="inverse_k"`: original call k=1, then residual scales
1/2, 1/3, etc. Numeric fixed damping remains supported. XSA applies no residual
damping; the experiment runner defaults non-damped modes to alpha=1.0.

## Use with the shared pipeline

```python
from functools import partial

from team_pipeline.runs import load_run
from team_pipeline.sampling import run_variant
from xsa_method import ScheduledBlocks, bind_runner

run_method = bind_runner(
    run_args,                    # object with .output, .repo, .protocol, .device, .checkpoint
    load_run=load_run,
    run_variant=partial(run_variant, schedule_cls=ScheduledBlocks),
    keep_checkpoints=False,
)
run_method("validation", "xsa_reuse", "member4_xsa_reuse", alpha=1.0)
```

`load_run(run_args)` returns `(root, protocol, config)` after provenance checks.
`run_variant` is the shared sampling and artifact function. These dependencies are
supplied explicitly; the package reads no notebook globals. `run_method(run_args,
split, method, name, load_run=..., run_variant=...)` is the unbound form.

`team_pipeline.methods.ScheduledBlocks` subclasses this package's scheduler and only
changes the CLI's default to a fixed `alpha=0.5`; with the same arguments both give
identical schedules. `team_pipeline/xsa.py` is a compatibility import of this
package's attention implementation.

## Tests and reproducibility

```bash
python -m pytest --import-mode=importlib -q tests_xsa
```

Tests use tiny CPU models and run the notebook's setup, import and configuration
cells in an IPython-free namespace with pip and git stubbed. They do not download
pretrained models. The notebook's test cell also runs these checks.

`method_sources()` returns a defensive copy of the source captured at package
import. `team_pipeline.provenance` combines it with a snapshot of the shared
pipeline modules to form the run fingerprint. After editing package files, restart
the kernel and prepare a new run. Runs prepared before the pipeline moved out of the
notebook cannot be resumed; reporting their saved artifacts remains available in
notebook section 12.

For Colab, make this package, `team_pipeline/`, the notebook and the tests available
in the same checkout. Setup supports a configurable Git ref; local unpushed edits
must be uploaded or pushed before a remote runtime can use them.
