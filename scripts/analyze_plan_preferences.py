"""Post-hoc readback of recorded CEM objectives; no new model or plant runs."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from scipy.stats import spearmanr
import yaml

def main(root,protocol_path,output):
    output.mkdir(parents=True,exist_ok=True)
    p=yaml.safe_load(protocol_path.read_text());s=p['evaluation']['mpc'];dt=p['evaluation']['control_dt_s']
    rows=[];hashes={}
    for complete in sorted((root/'D1').glob('*/complete.json')):
        metadata=json.loads(complete.read_text());trace=complete.parent/'full_trace.npz';hashes[metadata['run_id']]=hashlib.sha256(trace.read_bytes()).hexdigest()
        with np.load(trace,allow_pickle=False) as z:
            for cycle in range(metadata['cycles']):
                key=f'cycle_{cycle:03d}_';plans=z[key+'candidate_plans_W'];cost=z[key+'candidate_cost'].ravel()
                energy=plans.sum(axis=(-1,-2)).ravel()*dt;finite=np.isfinite(cost)
                corr=float(spearmanr(energy[finite],cost[finite]).statistic) if finite.sum()>2 and np.ptp(energy[finite])>0 and np.ptp(cost[finite])>0 else None
                plan=z[key+'plan_W'];field=z[key+'predicted_field_K'];reference=z[key+'reference_K'];held=z[key+'held_power_W']
                terms={'tracking_cost':s['tracking_weight']*(((field-reference[:,None,None])/s['temperature_scale_K'])**2).mean(),
                    'variance_cost':s['variance_weight']*(field.var(axis=(-2,-1))/s['temperature_scale_K']**2).mean(),
                    'slew_cost':s['slew_weight']*((np.diff(np.vstack([held,plan]),axis=0)/s['power_scale_W'])**2).mean(),
                    'energy_cost':s['energy_weight']*plan.sum()*dt/s['energy_scale_J']}
                selected_cost=float(cost[finite].min());reconstructed=float(sum(terms.values()))
                if not np.isclose(selected_cost,reconstructed,rtol=1e-7,atol=1e-9):raise ValueError('Recorded selected objective does not reconcile')
                rows.append({'run_id':metadata['run_id'],'variant':metadata['variant'],'model_seed':metadata['model_seed'],
                    'scenario_id':metadata['scenario_id'],'cycle':cycle,'selection_time_s':cycle*dt,
                    'finite_candidates':int(finite.sum()),'candidate_energy_cost_spearman':corr,
                    'selected_cost':selected_cost,'reconstructed_cost':reconstructed,
                    **{k:float(v) for k,v in terms.items()},'planned_energy_J':float(plan.sum()*dt)})
    with (output/'plan_objective_readback.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregate=[]
    for name in sorted({r['run_id'] for r in rows}):
        group=[r for r in rows if r['run_id']==name and r['selection_time_s']>=600]
        aggregate.append({'run_id':name,'late_cycles':len(group),**{k:float(np.mean([r[k] for r in group if r[k] is not None])) for k in ['candidate_energy_cost_spearman','tracking_cost','variance_cost','slew_cost','energy_cost','planned_energy_J']}})
    with (output/'plan_objective_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    (output/'method.json').write_text(json.dumps({'scope':'Unplanned post-hoc descriptive readback after D1; no new optimization, inference or plant integration.',
        'interpretation':'Correlation describes the learned/ROM total objective over recorded, correlated CEM candidates. It is not the sign of physical action response and does not establish a causal failure mechanism.',
        'late_cycle_rule':'selection time >=600 s, the fixed reference-ramp duration',
        'selected_objective_reconciliation':'all recorded minima match the sum of tracking, variance, slew and energy terms',
        'input_sha256':hashes,'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print(json.dumps({'cycles_reconciled':len(rows),'episodes':len(aggregate)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostic-dir',required=True,type=Path)
    p.add_argument('--protocol',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();main(a.diagnostic_dir,a.protocol,a.output)
