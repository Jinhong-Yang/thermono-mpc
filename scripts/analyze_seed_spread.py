"""Supplementary post-freeze analyses; existing aggregates remain unchanged."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--control-dir',required=True,type=Path)
parser.add_argument('--analysis-dir',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
def read(path):
    with path.open(encoding='utf-8') as f: return list(csv.DictReader(f))
def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
records=[]; energy=[]
for p in sorted(args.control_dir.glob('*_summary.json')):
    s=json.loads(p.read_text()); records.append(s)
    raw=read(p.with_name(p.name.replace('_summary.json','_raw.csv')))
    raw_energy=sum(float(r['energy_J']) for r in raw)
    raw_rmse=float(np.sqrt(np.mean([float(r['field_rmse_K'])**2 for r in raw])))
    assert abs(raw_energy-s['energy_J'])<1e-6 and abs(raw_rmse-s['rmse_K'])<1e-10
    energy.append({'run_id':s['run_id'],'controller':s['controller'],'seed':s['seed'],
        'scenario_id':s['scenario_id'],'category':s['category'],'energy_MJ':raw_energy/1e6,
        'rmse_K':raw_rmse,'almost_unheated_lt_0_2_MJ':raw_energy<200000,
        'source_raw':p.name.replace('_summary.json','_raw.csv')})
assert len(records)==120
write('energy_all_episodes.csv',energy)
low=[r for r in energy if r['almost_unheated_lt_0_2_MJ']]
write('low_energy_episodes.csv',low)
zone=[r for r in read(args.analysis_dir/'prediction_aggregate.csv') if r['metric']=='zone_rmse_K']
write('zone_forecast.csv',zone)
byseed=[]; spread=[]
for controller in ['B2_DATA_NO_MPC','B3_PINO_MPC']:
    for seed in [101,202,303]:
        group=[r for r in records if r['controller']==controller and r['seed']==seed]
        assert len(group)==12
        byseed.append({'controller':controller,'seed':seed,'n_scenarios':len(group),
            'mean_rmse_K':np.mean([r['rmse_K'] for r in group]),
            'mean_energy_MJ':np.mean([r['energy_J']/1e6 for r in group])})
    for metric in ['mean_rmse_K','mean_energy_MJ']:
        values=[r[metric] for r in byseed if r['controller']==controller]
        spread.append({'controller':controller,'metric':metric,'n_seeds':3,'mean':np.mean(values),
            'sample_sd_ddof1':np.std(values,ddof=1),'min':min(values),'max':max(values)})
write('control_by_seed.csv',byseed);write('control_seed_spread.csv',spread)
scenarios=sorted({r['scenario_id'] for r in records})
lookup={(r['controller'],r['seed'],r['scenario_id']):r['rmse_K'] for r in records}
fno=np.array([[lookup[('B2_DATA_NO_MPC',s,x)] for x in scenarios] for s in [101,202,303]])
pino=np.array([[lookup[('B3_PINO_MPC',s,x)] for x in scenarios] for s in [101,202,303]])
rom=np.array([lookup[('B1_ROM_CEM',None,x)] for x in scenarios])
rng=np.random.default_rng(12345); draws={'PINO-FNO':[],'PINO-ROM':[]}
for _ in range(10000):
    si=rng.integers(0,3,3); xi=rng.integers(0,12,12)
    # Same seed and scenario selections preserve paired methods and the crossed design.
    draws['PINO-FNO'].append((pino-fno)[np.ix_(si,xi)].mean())
    draws['PINO-ROM'].append((pino-rom[None,:])[np.ix_(si,xi)].mean())
boot=[]
for name,matrix in [('PINO-FNO',pino-fno),('PINO-ROM',pino-rom[None,:])]:
    lo,hi=np.percentile(draws[name],[2.5,97.5])
    boot.append({'contrast':name,'metric':'rmse_K','mean':matrix.mean(),'ci95_low':lo,'ci95_high':hi,
        'replicates':10000,'rng_seed':12345,'seeds':3,'scenarios':12,
        'method':'two-stage crossed seed/scenario paired percentile bootstrap; supplementary post-freeze'})
write('hier_bootstrap.csv',boot)
(OUT/'bootstrap_method.md').write_text(
    '# Supplementary bootstrap\n\nFor each of 10,000 replicates (NumPy seed 12345), three training-seed indices and twelve scenario indices were independently resampled with replacement. The resulting crossed seed-by-scenario submatrix used the same sampled indices for FNO and PINO, while ROM was repeated across the sampled seed index; its scenario pairing was preserved. Percentile intervals are exploratory supplementary intervals across both sources of variation and do not replace the frozen scenario-only intervals; cycles and prediction windows were not treated as independent replicates.\n',encoding='utf-8')
print(json.dumps({'raw_episodes_reconciled':120,'low_energy_episodes':len(low),'seed_spread':spread,'bootstrap':boot},indent=2))
