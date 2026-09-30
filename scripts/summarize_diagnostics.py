"""Aggregate D1/D2 at episode grain and retain numerical provenance."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from diagnostics import write_csv,write_json

def coverage_summary(values,lo,hi):
    a=np.asarray(values)
    return {'n':int(a.size),'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),
            'training_min':float(lo),'training_max':float(hi),'outside_training_range_fraction':float(np.mean((a<lo-1e-8)|(a>hi+1e-8)))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args()
    source=args.root/'D1';episodes=[json.loads(p.read_text()) for p in sorted(source.glob('*/complete.json'))]
    assert len(episodes)==21 and all(x['cycles']==90 for x in episodes),'D1 incomplete'
    write_csv(source/'episode_summary.csv',episodes)
    replay=[{'run_id':p.parent.name,**json.loads(p.read_text())} for p in sorted(source.glob('*/replay_comparison.json'))]
    write_csv(source/'replay_comparison.csv',replay)
    groups=[]
    for variant in ['ROM','data_only','physics_informed']:
        group=[r for r in episodes if r['variant']==variant]
        groups.append({'variant':variant,'episodes':len(group),
            **{k:float(np.mean([r[k] for r in group])) for k in ['closed_loop_rmse_K','closed_loop_energy_J','end_mean_bias_K',
                'end_field_rmse_K','end_zone_rmse_K','oracle_end_mean_bias_K','oracle_end_field_rmse_K','projected_state_rmse_K']},
            'episodes_bias_above_1_K':sum(r['end_mean_bias_K']>1 for r in group),
            'episodes_oracle_bias_above_1_K':sum(r['oracle_end_mean_bias_K']>1 for r in group),
            'episodes_end_field_rmse_at_most_1_K':sum(r['end_field_rmse_K']<=1 for r in group)})
    write_csv(source/'family_summary.csv',groups)
    rom=groups[0];neural=groups[1:]
    if all(g['episodes_bias_above_1_K']>=7 and g['episodes_oracle_bias_above_1_K']>=7 and
           abs(g['end_mean_bias_K'])>abs(rom['end_mean_bias_K']) for g in neural):branch='a'
    elif all(g['episodes_end_field_rmse_at_most_1_K']>=7 and g['closed_loop_rmse_K']>rom['closed_loop_rmse_K'] for g in neural):branch='b'
    else:branch='c'
    write_json(source/'branch_result.json',{'branch':branch,'criteria_file':'configs/diagnostics/D1.yaml','families':groups,
        'note':'Episode-level descriptive criteria, chosen before final runs. No time-sample significance test. Author review required before manuscript narrative.'})
    if branch=='b':write_json(args.root/'STOP_REPORT.json',{'reason':'D1 branch b: accurate forecasts with poor tracking; stop manuscript narrative and publication','source':'D1/branch_result.json'})

    manifest=json.loads(args.manifest.read_text());train_power=[];train_slew=[];train_energy=[]
    for item in manifest['trajectories']:
        if item['split']!='train':continue
        with np.load(args.manifest.parent/item['file'],allow_pickle=False) as z:power=z['power_W']
        train_power.append(power);train_slew.append(np.diff(np.vstack([np.zeros(power.shape[1]),power]),axis=0))
        train_energy.extend([power[t:t+12].sum()*10 for t in range(len(power)-11)])
    tp=np.vstack(train_power);ts=np.vstack(train_slew);te=np.asarray(train_energy)
    rows=[];hist=[];energyhist=[]
    def add(name,run,power,slew,energy):
        for metric,a,b in [('power_W',power,tp),('slew_W',slew,ts)]:
            for zone in range(a.shape[1]):
                rows.append({'run_id':run,'population':name,'metric':metric,'zone':zone,**coverage_summary(a[:,zone],b[:,zone].min(),b[:,zone].max())})
                bins=np.linspace(0,2000,21) if metric=='power_W' else np.linspace(-400,400,17)
                assert a[:,zone].min()>=bins[0]-1e-8 and a[:,zone].max()<=bins[-1]+1e-8
                counts,edges=np.histogram(np.clip(a[:,zone],bins[0],bins[-1]),bins=bins)
                assert counts.sum()==len(a)
                hist.extend({'run_id':run,'population':name,'metric':metric,'zone':zone,'left':float(edges[i]),'right':float(edges[i+1]),'count':int(n)} for i,n in enumerate(counts))
        rows.append({'run_id':run,'population':name,'metric':'plan_energy_J','zone':'all',**coverage_summary(energy,te.min(),te.max())})
        counts,edges=np.histogram(energy,bins=np.linspace(0,720000,25))
        energyhist.extend({'run_id':run,'population':name,'left_J':float(edges[i]),'right_J':float(edges[i+1]),'count':int(n)} for i,n in enumerate(counts))
    add('training','training_24_trajectories',tp,ts,te)
    for ep in episodes:
        run=ep['run_id'];collected={k:{'power':[],'slew':[],'energy':[]} for k in ['selected','candidate','elite']}
        with np.load(source/run/'full_trace.npz',allow_pickle=False) as z:
            for cycle in range(ep['cycles']):
                prefix=f'cycle_{cycle:03d}_';plans=z[prefix+'candidate_plans_W'];elites=z[prefix+'elite_indices'];held=z[prefix+'held_power_W']
                eliteplans=np.concatenate([plans[i,elites[i]] for i in range(len(plans))])
                populations={'selected':z[prefix+'plan_W'][None],'candidate':plans.reshape(-1,12,3),'elite':eliteplans}
                for name,a in populations.items():
                    slew=np.diff(np.concatenate([np.broadcast_to(held,(len(a),1,3)),a],axis=1),axis=1)
                    collected[name]['power'].append(a.reshape(-1,3));collected[name]['slew'].append(slew.reshape(-1,3));collected[name]['energy'].append(a.sum(axis=(1,2))*10)
        for name,a in collected.items():add(name,run,np.vstack(a['power']),np.vstack(a['slew']),np.concatenate(a['energy']))
    out=args.root/'D2';write_csv(out/'coverage.csv',rows);write_csv(out/'histograms.csv',hist);write_csv(out/'energy_histograms.csv',energyhist)
    write_json(out/'complete.json',{'d1_episodes':len(episodes),'training_trajectories':24,'scope':'descriptive coverage; correlated candidate occurrences, not independent samples','floating_point_range_tolerance':1e-8,'rows':len(rows)})
    print(json.dumps({'D1_branch':branch,'families':groups,'D2_rows':len(rows)},indent=2))

if __name__=='__main__':main()
