"""Exercise the notebook's real reference export without downloading Imagenette."""
import ast
import hashlib
import json
from pathlib import Path
import warnings

from PIL import Image
import pytest

ROOT = Path(__file__).parents[1]


def prepare_function():
    nb = json.loads((ROOT / 'DiT_Layer_Looping_Damped_colab.ipynb').read_text())
    source = next(''.join(c['source']) for c in nb['cells']
                  if 'def prepare_imagenette_reference(' in ''.join(c['source']))
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'prepare_imagenette_reference')
    ns = dict(Path=Path, json=json, hashlib=hashlib, warnings=warnings,
              validate_pairs=lambda pairs: pairs,
              IMAGENETTE_CLASSES=[(0, 'n03394916', 'French horn')],
              write_json=lambda p, v: p.write_text(json.dumps(v)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), ns)
    return ns['prepare_imagenette_reference']


def dataset(tmp_path, monkeypatch, valid):
    import torchvision.datasets
    # The external download is unnecessary for this tiny on-disk dataset.
    monkeypatch.setattr(torchvision.datasets, 'Imagenette', lambda **kwargs: None)
    folder = tmp_path / 'source/imagenette2/train/n03394916'
    folder.mkdir(parents=True)
    (folder / 'empty.JPEG').write_bytes(b'')
    (folder / 'invalid.JPEG').write_bytes(b'not a jpeg')
    for i in range(valid):
        Image.new('RGB', (12, 16), (i * 30, 0, 0)).save(folder / f'valid{i}.JPEG')
    return tmp_path / 'source'


def test_replaces_unreadable_images_and_reuses_balanced_reference(tmp_path, monkeypatch):
    source = dataset(tmp_path, monkeypatch, valid=3)
    prepare = prepare_function()
    with pytest.warns(UserWarning, match='Skipping unreadable'):
        output, description = prepare(source, tmp_path / 'prepared', [[0, 1]],
                                      images_per_class=3, image_size=8)
    manifest = json.loads((output / 'reference_manifest.json').read_text())
    assert manifest['images'] == 3
    assert len(manifest['skipped_sources']) == 2
    assert all('valid' in row['source'] for row in manifest['inventory'])
    for path in output.glob('*.png'):
        with Image.open(path) as image:
            image.load()
            assert image.size == (8, 8)
    cached, _ = prepare(source, tmp_path / 'prepared', [[0, 1]], images_per_class=3, image_size=8)
    assert cached == output
    with pytest.warns(UserWarning):
        repeated, _ = prepare(source, tmp_path / 'other', [[0, 1]], images_per_class=3, image_size=8)
    assert json.loads((repeated / 'reference_manifest.json').read_text()) == manifest


def test_insufficient_readable_images_does_not_publish_manifest(tmp_path, monkeypatch):
    source = dataset(tmp_path, monkeypatch, valid=1)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match='readable images'):
        prepare_function()(source, tmp_path / 'prepared', [[0, 1]], images_per_class=3)
    assert not list((tmp_path / 'prepared').rglob('reference_manifest.json'))
