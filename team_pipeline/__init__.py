"""Shared DiT experiment pipeline: protocol, sampling, evaluation, metrics and reports.

Importing this package sets the environment PyTorch reads when it first initializes
cuBLAS / MPS kernels. It is the only place these are set; existing values win.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
