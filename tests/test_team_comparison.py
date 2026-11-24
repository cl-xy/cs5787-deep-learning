"""Exercise comparison on real saved tensors, without model downloads."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
try:
    import torch
    import numpy as np
    import pandas as pd
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires torch, numpy and pandas')
class TeamComparisonTests(unittest.TestCase):
    def test_common_baseline_recovery_and_reject_mismatched_design(self):
        nb=json.loads((Path(__file__).parents[1]/'DiT_Layer_Looping_Damped_colab.ipynb').read_text())
        ns=dict(Path=Path,json=json,torch=torch,np=np,pd=pd,display=lambda x:None,
                preprocess_image=lambda x,size:x,write_json=lambda p,v:p.write_text(json.dumps(v)))
        for c in nb['cells']:
            if c.get('id') not in {'pairing-definitions','team-comparison-definitions'}: continue
            for node in ast.parse(''.join(c['source'])).body:
                if isinstance(node,ast.FunctionDef):
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook>','exec'),ns)
        specs=ns['pairing_specs'](10)
        config={'image_size':2,'checkpoint_sha256':'weights','protocol_sha256':'protocol'}
        pairs=[[0,10],[0,11]]
        class Distance:
            def __call__(self,a,b): return (a-b).abs().mean()
        with TemporaryDirectory() as tmp:
            roots={mode:Path(tmp)/mode for mode in ('damped','pruned')}
            for mode,root in roots.items():
                base=root/'pairing_experiments'; split=base/'validation';split.mkdir(parents=True)
                design={'anchor':10,'specs':specs,'contract':{'mode':mode,'source_sha256':'code','config':config}}
                (base/'design.json').write_text(json.dumps(design))
                fid=dict(implementation='clean-fid',mode='clean',features='inception',reference_images=5000,
                         generated_images=2,reference_inventory_sha256='reference')
                (split/'evaluation_extended.json').write_text(json.dumps({'fid':fid,'packages':{}}))
                rows=[]
                for spec in [None,*specs]:
                    name='original' if spec is None else spec['name']+'_'+mode
                    folder=split/name;folder.mkdir()
                    settings=None if spec is None else dict(removed=spec['removed'],looped=spec['looped'] if mode=='damped' else [])
                    manifest=dict(pairs=pairs,config=config,runtime={},method_metadata={'settings':settings})
                    (folder/'manifest.json').write_text(json.dumps(manifest))
                    pd.DataFrame([(i,*p) for i,p in enumerate(pairs)],columns=['index','label','seed']).to_csv(folder/'samples.csv',index=False)
                    # Deliberately different local originals: comparison must use damped original for both.
                    value=(0. if mode=='damped' else 1.) if spec is None else (.25 if mode=='damped' else .75)
                    for i in range(2):torch.save(torch.full((1,3,2,2),value),folder/f'image_{i:04d}.pt')
                    rows.append(dict(variant=name,images=2,fid=3. if mode=='damped' else 4.))
                pd.DataFrame(rows).to_csv(split/'results_extended.csv',index=False)
            result=ns['compare_team_pairings'](roots['damped'],roots['pruned'],metric=Distance())
            self.assertEqual(len(result),5)
            self.assertTrue((result['lpips_mean_recovered']==.5).all())
            self.assertTrue((result['fid_recovered']==1.).all())
            output=roots['damped']/'pairing_experiments/validation/team_comparison'
            self.assertEqual(len(pd.read_csv(output/'paired_recovery.csv')),10)
            path=roots['pruned']/'pairing_experiments/design.json'
            bad=json.loads(path.read_text());bad['anchor']=11;path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                ns['compare_team_pairings'](roots['damped'],roots['pruned'],metric=Distance())


if __name__=='__main__':unittest.main()
