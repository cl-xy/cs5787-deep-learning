"""Small real-tensor tests; no weights, torchvision or inference downloads."""
import ast
from contextlib import contextmanager
import json
import math
from pathlib import Path
import unittest

try:
    import torch
    from torch import nn
    import numpy as np
    import pandas as pd
except ImportError:
    torch = None


def load_definitions():
    notebook = json.loads((Path(__file__).parents[1] / 'DiT_Layer_Looping_colab.ipynb').read_text())
    ns = dict(torch=torch, nn=nn, np=np, pd=pd, math=math, contextmanager=contextmanager)
    wanted = {'ScheduledBlocks','PairingBlocks','pairing_schedule','pairing_specs',
              'select_pairing_anchor','pairing_attention_capture'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), ns)
    return ns


@unittest.skipIf(torch is None, 'Requires torch, pandas and numpy')
class PairingTensorTests(unittest.TestCase):
    def test_damped_update_and_unique_parameter_registration(self):
        ns = load_definitions()
        class Multiply(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(float(value)))
            def forward(self, x, c):
                return x * self.weight
        blocks = [Multiply(i) for i in (2,3,4,5)]
        loop = ns['PairingBlocks'](blocks, [3], [2])
        x = torch.ones(1)
        for block in loop:
            x = block(x, None)
        torch.testing.assert_close(x, torch.tensor([60.]))
        self.assertEqual(len(list(loop.parameters())), 3)
        self.assertEqual(len(loop.state_dict()), 3)
        prune = ns['PairingBlocks'](blocks, [3], [])
        x = torch.ones(1)
        for block in prune:
            x = block(x, None)
        torch.testing.assert_close(x, torch.tensor([24.]))

    def test_selection_reverses_ranking_and_protects_anchor_endpoints(self):
        ns = load_definitions()
        rows = [dict(block=i, timestep=t, mean_cfg=(100 if i in (0,27) else i))
                for i in range(28) for t in (999,500,0)]
        anchor, scores = ns['select_pairing_anchor'](pd.DataFrame(rows))
        self.assertEqual(anchor, 26)
        rows = [r for r in rows if not (r['block']==4 and r['timestep']==0)]
        with self.assertRaises(ValueError):
            ns['select_pairing_anchor'](pd.DataFrame(rows))

    def test_attention_capture_replay_conditional_branch_and_cleanup(self):
        ns = load_definitions()
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads, self.scale = 1, 1.
                self.qkv = nn.Linear(1,3,bias=False)
                nn.init.ones_(self.qkv.weight)
            def forward(self,x):
                return x
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()
            def forward(self,x,c):
                return self.attn(x)
        model = nn.Module()
        model.t_embedder = nn.Identity()
        model.blocks = ns['PairingBlocks']([Block(),Block()], [1], [0])
        x = torch.tensor([[[0.],[1.],[2.],[3.]], [[9.],[8.],[7.],[6.]]])
        with ns['pairing_attention_capture'](model,[0],1) as records:
            for t in (999,500):
                model.t_embedder(torch.tensor([t,t]))
                for block in model.blocks:
                    block(x,None)
        self.assertEqual(set(records), {(0,1),(0,2)})
        expected = (x[:1] @ x[:1].transpose(-2,-1)).softmax(-1).mean((0,1)).reshape(2,2)
        torch.testing.assert_close(records[(0,2)]['map'], expected)
        self.assertEqual(records[(0,2)]['timestep'], 500)
        self.assertFalse(model.t_embedder._forward_pre_hooks)
        self.assertFalse(model.blocks.retained['0'].attn._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
