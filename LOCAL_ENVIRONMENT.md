# Local macOS environment

This project uses `.venv` with Homebrew Python 3.12 and pip-installed packages.
It is separate from the Conda `lumiq_env` environment that crashed during
OpenMP initialization. `.venv` is excluded from Git.

PyTorch is pinned to 2.14.1 with torchvision 0.29.1 for macOS 14 MPS support.
After installing or changing these packages, restart the notebook kernel.

## Use in VS Code

Open a notebook, click **Select Kernel** in the top right, then choose
**Python Environments** and the project's `.venv/bin/python` (Python 3.12).
If listed under Jupyter kernels, its display name is **CS5787 DiT (.venv)**.
An already open notebook may retain its previous kernel until you select this one.

To confirm in a notebook cell:

```python
import sys, torch
print(sys.executable)  # should end in this project's .venv/bin/python
print(torch.__version__)
```

For terminal work, from the project root:

```sh
source .venv/bin/activate
```

## Recreate on an Apple Silicon Mac

From the project root, with Homebrew Python 3.12 installed:

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.lock.txt
.venv/bin/python -m ipykernel install --sys-prefix --name cs5787-dit --display-name 'CS5787 DiT (.venv)'
.venv/bin/python -m pip check
```

`requirements-local.txt` lists direct dependencies;
`requirements-local.lock.txt` records the complete installed versions.
These dependencies cover the damped/pruned notebooks and FID evaluation.
Optional TPIPS evaluation is not installed. Model weights and datasets are
downloaded separately when needed. The upstream CUDA environment file is for
NVIDIA systems, not this Mac.
