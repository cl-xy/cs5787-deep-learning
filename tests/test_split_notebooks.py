"""Standalone notebook assignment and shared design contracts."""
import ast
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).parents[1]


def notebook(mode):
    path = ROOT / f'DiT_Layer_Looping_{mode.title()}_colab.ipynb'
    assert path.is_file(), f'Missing standalone {mode} notebook'
    return json.loads(path.read_text())


def functions(nb):
    ns = {'Path': Path, 'json': json}
    source = ''.join(next(c for c in nb['cells'] if c.get('id') == 'pairing-definitions')['source'])
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            exec(compile(ast.Module(body=[node],type_ignores=[]), '<split-notebook>', 'exec'), ns)
    return ns


@pytest.mark.parametrize('mode', ['damped','pruned'])
def test_runner_generates_all_five_pairings_only_in_assigned_mode(mode,tmp_path):
    ns = functions(notebook(mode))
    specs = ns['pairing_specs'](10)
    assert len(specs) == 5
    design = {'specs': specs, 'contract': {'source_sha256':'hash'}}
    ns.update(PAIRING_MODE=mode, KEEP_CHECKPOINTS=False,
              prepare_pairing_experiments=lambda: (tmp_path,{'splits':{'validation':[[0,10]]}}, {}, design),
              pairing_complete=lambda *args: False)
    def run_variant(config,pairs,folder,**kwargs):
        folder.mkdir(parents=True)
        (folder/'metadata.json').write_text(json.dumps(kwargs['method_metadata']))
    ns['run_variant'] = run_variant
    root,names,_ = ns['run_pairing_experiments']()
    assert names == ['original']+[s['name']+'_'+mode for s in specs]
    assert len(list(root.iterdir())) == 6
    for spec in specs:
        meta=json.loads((root/(spec['name']+'_'+mode)/'metadata.json').read_text())
        settings=meta['settings']
        assert settings['removed']==spec['removed']
        assert settings['looped']==(spec['looped'] if mode=='damped' else [])
        assert len(ns['pairing_schedule'](28,**settings)) == (28 if mode=='damped' else 28-len(spec['removed']))


def test_shared_code_protocol_and_specs_are_identical():
    original = json.loads((ROOT/'DiT_Layer_Looping_colab.ipynb').read_text())
    shared_ids = {c['id']:c['source'] for c in original['cells'][1:27] if c['cell_type']=='code'}
    specs=[]
    for mode in ('damped','pruned'):
        nb = notebook(mode)
        specs.append(functions(nb)['pairing_specs'](10))
        for cell in nb['cells']:
            if cell['id'] in shared_ids:
                assert cell['source'] == shared_ids[cell['id']]
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
        assert shared_ids.keys() <= {c['id'] for c in nb['cells']}
    assert specs[0] == specs[1]


def test_shared_design_import_works_across_devices_without_calibration(tmp_path):
    ns=functions(notebook('pruned'))
    config={'device':'cuda:0','checkpoint':'/local/model.pt','checkpoint_sha256':'weights','protocol_sha256':'protocol'}
    portable={**config,'device':'cuda:1','checkpoint':'/teammate/model.pt'}
    design={'anchor':10,'specs':ns['pairing_specs'](10), 'contract':{
        'mode':'damped','config':portable,'source_sha256':'code','calibration_pairs':[[0,1]]}}
    shared=tmp_path/'shared.json'; shared.write_text(json.dumps(design))
    ns.update(PAIRING_MODE='pruned', SHARED_PAIRING_DESIGN=shared,RUN_ARGS=None,
              load_run=lambda _: (tmp_path,{'splits':{'calibration':[[0,1]]}},config),
              pairing_source_hash=lambda:'code',write_json=lambda p,v:p.write_text(json.dumps(v)))
    _,_,_,imported=ns['prepare_pairing_experiments']()
    assert imported['anchor']==10
    assert imported['contract']['mode']=='pruned'
    assert not (tmp_path/'pairing_experiments/profile').exists()


@pytest.mark.parametrize('mode', ['damped','pruned'])
def test_full_test_cell_uses_frozen_5000_image_split(mode,tmp_path):
    from types import SimpleNamespace
    nb=notebook(mode)
    cells=[c for c in nb['cells'] if c.get('id')=='pairing-test-run']
    assert cells, 'Missing full held-out test cell'
    pairs=[[label,label*500+i] for label in range(10) for i in range(500)]
    (tmp_path/'protocol.json').write_text(json.dumps({'splits':{'test':pairs}}))
    calls=[]
    ns=dict(Path=Path,json=json,RUN_ARGS=SimpleNamespace(output=tmp_path),CLASS_LABELS=tuple(range(10)))
    def run(split):
        calls.append(split)
        return tmp_path/split,['original']+[f'variant_{i}_{mode}' for i in range(5)],{}
    ns['run_pairing_experiments']=run
    exec(''.join(cells[0]['source']),ns)
    assert calls==['test']
    assert ns['PAIRING_SPLIT']=='test'
    assert len(ns['pairing_test_pairs'])==5000
    (tmp_path/'protocol.json').write_text(json.dumps({'splits':{'test':pairs[:20]}}))
    calls.clear()
    with pytest.raises(ValueError):
        exec(''.join(cells[0]['source']),ns)
    assert calls==[]
