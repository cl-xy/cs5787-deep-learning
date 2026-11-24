"""Dependency-free checks of the notebook's experiment design and execution plan."""
import ast
import json
from pathlib import Path
import pytest


def definitions():
    nb = json.loads((Path(__file__).parents[1] / 'DiT_Layer_Looping_colab.ipynb').read_text())
    cells = [c for c in nb['cells'] if c.get('id') == 'pairing-definitions']
    assert cells, 'Pairing experiment definitions are missing'
    tree = ast.parse(''.join(cells[0]['source']))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'pairing_schedule', 'pairing_specs'}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<pairing>', 'exec'), ns)
    return ns


def test_before_and_after_preserve_depth_and_repeat_anchor_after_first_pass():
    ns = definitions()
    specs = ns['pairing_specs'](10)
    assert len(specs) == 5
    assert specs[0]['removed'] == [9]
    assert specs[1]['removed'] == [11]
    for spec in specs:
        plan = ns['pairing_schedule'](28, spec['removed'], spec['looped'])
        assert len(plan) == 28
        assert len({i for i, _, _ in plan}) == 28 - len(spec['removed'])
        assert all(i not in spec['removed'] for i, _, _ in plan)
        assert [(i, a) for i, a, repeat in plan if repeat] == [(i, .5) for i in spec['looped']]
        prune = ns['pairing_schedule'](28, spec['removed'], [])
        assert len(prune) == 28 - len(spec['removed'])
    plan = ns['pairing_schedule'](6, [4, 5], [2, 3])
    assert plan == [(0, 1., False), (1, 1., False), (2, 1., False), (3, 1., False), (2, .5, True), (3, .5, True)]


@pytest.mark.parametrize('removed,looped', [([2],[2]), ([2],[0,1]), ([7],[1]), ([2,2],[1,3]), ([2,4],[0,1]), ([3,4],[2,1])])
def test_invalid_pairings_fail_before_model_loading(removed, looped):
    with pytest.raises(ValueError):
        definitions()['pairing_schedule'](6, removed, looped)
