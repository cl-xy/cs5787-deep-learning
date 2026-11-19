"""Restore compact DiT checkpoints, including XSA schedule metadata."""
import torch

from .schedule import ScheduledBlocks


def load_compact_checkpoint(path):
    from models import DiT_XL_2
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    meta = payload["schedule"]
    if meta.get("mode") == "custom":
        raise ValueError("Custom checkpoints require the corresponding method builder before loading state_dict")
    if meta.get("index_base", 0) != 0:
        raise ValueError("Expected zero-based block indices")
    with torch.device("meta"):
        model = DiT_XL_2(input_size=32, num_classes=1000)
    model.blocks = ScheduledBlocks(model.blocks, meta["removed"], meta["mode"],
                                   meta["alpha"], xsa_eps=meta.get("xsa_eps", 1e-8))
    if "retained" in meta and meta["retained"] != model.blocks.metadata()["retained"]:
        raise ValueError("Checkpoint original-index mapping is inconsistent")
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    return model.eval().requires_grad_(False)


