"""D9/D10 reviewer diagnostics. Writes only new post-freeze directories.

D9 compares plans from identical estimated/true starting states and previous
commands. Counterfactual ROM plans are reselected at the neural trajectory,
not borrowed from a different closed-loop trajectory.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import yaml
from scipy.optimize import minimize
from thermono_mpc.controllers import CEMMPC, ROMPredictor, MPCConfig, MPCResult, limit_power
from thermono_mpc.process import ProcessConfig, ThermalPlant, ThermalState
from thermono_mpc.simulation import Scenario, run_episode, make_predictor
from diagnostics import write_csv, write_json, episode_kwargs, replay_compare

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/diagnostics_v3'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def setup(diag):
    p=yaml.safe_load((ROOT/'configs/frozen/protocol_v2.yaml').read_text())
    cfg=yaml.safe_load((ROOT/f'configs/diagnostics/{diag}.yaml').read_text())
    assert sha(ROOT/'configs/frozen/protocol_v2.yaml')==cfg['base_protocol_sha256']
    dest=OUT/diag;dest.mkdir(parents=True,exist_ok=True)
    snapshot=dest/'executed_source';snapshot.mkdir(exist_ok=True)
    files=[Path(__file__),ROOT/'scripts/diagnostics.py',ROOT/f'configs/diagnostics/{diag}.yaml',ROOT/'configs/frozen/protocol_v2.yaml']+list((ROOT/'src/thermono_mpc').glob('*.py'))
    manifest=[]
    for f in files:
        q=snapshot/f.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists() and q.read_bytes()!=f.read_bytes():raise RuntimeError('Executed snapshot differs; use a new diagnostic directory')
        q.write_bytes(f.read_bytes());manifest.append({'path':f.relative_to(ROOT).as_posix(),'sha256':sha(f)})
    write_json(dest/'execution.json',{'scope':'post-freeze','diagnostic':diag,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'source_files':manifest,'started_unix':time.time(),'configuration':cfg})
    return p,cfg,dest

def costs(fields, seq, ref, previous, s, dt=10.):
    fields=np.asarray(fields,dtype=np.float64)
    delta=np.diff(np.concatenate([np.broadcast_to(previous,(len(seq),1,len(previous))),seq],axis=1),axis=1)
    return (s.tracking_weight*(((fields-ref[None,:,None,None])/s.temperature_scale_K)**2).mean(axis=(1,2,3))+
        s.variance_weight*(fields.var(axis=(-2,-1))/s.temperature_scale_K**2).mean(axis=1)+
        s.slew_weight*((delta/s.power_scale_W)**2).mean(axis=(1,2))+
        s.energy_weight*seq.sum(axis=(1,2))*dt/s.energy_scale_J)

def d9_job(job):
    import torch
    torch.set_num_threads(1)
    path,device=job;path=Path(path);tic=time.perf_counter()
    protocol=yaml.safe_load((ROOT/'configs/frozen/protocol_v2.yaml').read_text())
    config=ProcessConfig(**protocol['process']).with_grid(*protocol['evaluation']['grid'])
    settings=MPCConfig(**protocol['evaluation']['mpc'])
    meta=json.loads((path/'complete.json').read_text());scenario=Scenario(**next(x for x in protocol['evaluation']['scenarios'] if x['scenario_id']==meta['scenario_id']))
    actual=replace(config,rho=config.rho*scenario.rho_scale,cp=config.cp*scenario.cp_scale,k=config.k*scenario.k_scale)
    plant=ThermalPlant(actual);rom=ROMPredictor(config)
    neural=meta['model_seed'] is not None
    predictor=make_predictor(config,ROOT/f"artifacts/generated/protocol_v1/{meta['variant']}_seed{meta['model_seed']}.json",device) if neural else rom
    dest=OUT/'D9'/meta['run_id'];dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return json.loads((dest/'complete.json').read_text())
    assert sha(path/'full_trace.npz')==meta['full_trace_sha256']
    replay=json.loads((path/'replay_comparison.json').read_text())
    assert all(v==0 for k,v in replay.items() if k.endswith('max_abs_diff'))
    rejected_energy=[];accepted_energy=[];rows=[];cross=[];probe=[];arrays={}
    with np.load(path/'full_trace.npz',allow_pickle=False) as z:
        for cycle in range(meta['cycles']):
            pref=f'cycle_{cycle:03d}_';get=lambda key:z[pref+key]
            candidates=get('candidate_plans_W');saved_cost=get('candidate_cost')
            # CEM validates finite predictions and sets cost=inf only on thermal-limit rejection.
            reject=~np.isfinite(saved_cost);energy=candidates.sum(axis=(2,3))*10/1e6
            rejected_energy.extend(energy[reject].tolist());accepted_energy.extend(energy[~reject].tolist())
            own=get('plan_W');best=np.unravel_index(np.argmin(saved_cost),saved_cost.shape)
            assert np.array_equal(own,candidates[best]),(meta['run_id'],cycle)
            row={'run_id':meta['run_id'],'cycle':cycle,'candidates':int(reject.size),'thermal_rejected':int(reject.sum()),'rejected_fraction':float(reject.mean()),'selected_energy_MJ':float(own.sum()*10/1e6)}
            rows.append(row)
            if not neural:continue
            field,zones,held,ref=[get(k) for k in ['projected_field_K','projected_zone_K','held_power_W','reference_K']]
            # Independent deterministic CEM search, same state, horizon and actuator history.
            selector=CEMMPC(config,rom,settings,seed=scenario.seed+100000+cycle)
            alternative=selector.optimize(field,zones,ref,held,10.).sequence_W
            plans=np.stack([own,alternative]);pred=predictor.predict(field,zones,plans,10.)
            pcost=costs(pred.temperature_K,plans,ref,held,settings)
            rpred=rom.predict(field,zones,plans,10.);rcost=costs(rpred.temperature_K,plans,ref,held,settings)
            true=ThermalState(get('aligned_true_field_K'),get('aligned_true_zone_K'))
            realized=[]
            for power in alternative:
                true=plant.advance(true,power,10.,2.);realized.append(true.field.copy())
            realized=np.stack(realized)
            tc=costs(np.stack([get('realized_field_K'),realized]),plans,ref,held,settings)
            pred_feasible=(pred.temperature_K.max(axis=(1,2,3))<=config.solid_limit)&(pred.zone_temperature_K.max(axis=(1,2))<=config.zone_limit)
            cross.append({'run_id':meta['run_id'],'cycle':cycle,'neural_own_cost':float(pcost[0]),'neural_rom_cost':float(pcost[1]),'neural_difference_own_minus_rom':float(pcost[0]-pcost[1]),'rom_difference_own_minus_rom':float(rcost[0]-rcost[1]),'plant_own_cost':float(tc[0]),'plant_rom_cost':float(tc[1]),'plant_difference_own_minus_rom':float(tc[0]-tc[1]),'neural_prefers_own':bool(pcost[0]<pcost[1]),'plant_prefers_rom':bool(tc[0]>tc[1]),'strict_reversal':bool(pcost[0]<pcost[1] and tc[0]>tc[1]),'neural_own_feasible':bool(pred_feasible[0]),'neural_rom_feasible':bool(pred_feasible[1]),'rom_plan_energy_MJ':float(alternative.sum()*10/1e6)})
            probe_plans=[]
            for target in [0,500,1000,2000]:
                prev=held.copy();seq=[]
                for _ in range(settings.horizon):
                    prev=limit_power(np.full(config.zones,target),prev,config);seq.append(prev.copy())
                probe_plans.append(seq)
            pp=np.array(probe_plans);response=predictor.predict(field,zones,pp,10.).temperature_K[:,-1].mean(axis=(1,2))
            pr={'run_id':meta['run_id'],'cycle':cycle,'nonmonotone':bool(np.any(np.diff(response)<-1e-5))}
            pr.update({f'end_mean_{target}W_K':float(val) for target,val in zip([0,500,1000,2000],response)});probe.append(pr)
            arrays[pref+'rom_plan_W']=alternative;arrays[pref+'rom_plan_realized_field_K']=realized
            arrays[pref+'neural_objectives']=pcost;arrays[pref+'plant_objectives']=tc;arrays[pref+'probe_endpoint_means_K']=response
            if cycle%15==0:print(meta['run_id'],cycle,flush=True)
    write_csv(dest/'screening.csv',rows);write_csv(dest/'cross_evaluation.csv',cross);write_csv(dest/'response_probes.csv',probe)
    np.savez_compressed(dest/'counterfactuals.npz',**arrays)
    summary={'run_id':meta['run_id'],'scenario':scenario.scenario_id,'variant':meta['variant'],'seed':meta['model_seed'],'n_calls':len(rows),'n_candidates':sum(r['candidates'] for r in rows),'rejected_fraction':float(np.mean([r['rejected_fraction'] for r in rows])),'median_rejected_energy_MJ':float(np.median(rejected_energy)) if rejected_energy else None,'median_accepted_energy_MJ':float(np.median(accepted_energy)) if accepted_energy else None,'n_cross':len(cross),'neural_prefers_own_fraction':float(np.mean([r['neural_prefers_own'] for r in cross])) if cross else None,'plant_prefers_rom_fraction':float(np.mean([r['plant_prefers_rom'] for r in cross])) if cross else None,'strict_reversal_fraction':float(np.mean([r['strict_reversal'] for r in cross])) if cross else None,'nonmonotone_fraction':float(np.mean([r['nonmonotone'] for r in probe])) if probe else None,'input_sha256':sha(path/'full_trace.npz'),'selected_plan_exact_equality':True,'original_replay_equality':replay,'wall_s':time.perf_counter()-tic}
    write_json(dest/'complete.json',summary);return summary

class ScaledSingleStart:
    """D10 intervention: normalized decision variables, one repaired warm start.

    Preserves the physical objective and constraints. The frozen implementation
    remains unchanged and available as SLSQPMPC.
    """
    def __init__(self,base):
        self.process,self.settings,self.predictor=base.process,base.settings,base.predictor
        self.warm=None;self.calls=[]
    def optimize(self,field_K,zones_K,reference_K,previous_W,dt_s):
        s,c=self.settings,self.process;idx=np.minimum(np.arange(s.horizon)*s.blocks//s.horizon,s.blocks-1)
        scale=np.tile(c.pmax,s.blocks)
        def sequence(v):return (v*scale).reshape(s.blocks,c.zones)[idx]
        def objective(v):
            seq=sequence(v);p=self.predictor.predict(field_K,zones_K,seq[None],dt_s)
            return float(costs(p.temperature_K,seq[None],reference_K,previous_W,s,dt_s)[0])
        def slew(v):
            delta=np.diff(np.vstack([previous_W,(v*scale).reshape(s.blocks,c.zones)]),axis=0)
            return np.r_[(np.asarray(c.slew)-delta).ravel(),(np.asarray(c.slew)+delta).ravel()]
        def thermal(v):
            p=self.predictor.predict(field_K,zones_K,sequence(v)[None],dt_s)
            return np.r_[c.solid_limit-p.temperature_K[0].max(axis=(1,2)),c.zone_limit-p.zone_temperature_K[0].max(axis=1)]
        guess=np.broadcast_to(np.minimum(previous_W+c.slew,c.pmax),(s.blocks,c.zones)).copy() if self.warm is None else self.warm[np.minimum(np.arange(s.blocks)*s.horizon//s.blocks+1,s.horizon-1)].copy()
        for j in range(s.blocks):guess[j]=limit_power(guess[j],previous_W if j==0 else guess[j-1],c)
        x0=guess.ravel()/scale
        r=minimize(objective,x0,method='SLSQP',bounds=[(0,1)]*len(x0),constraints=[{'type':'ineq','fun':slew},{'type':'ineq','fun':thermal}],options={'maxiter':80,'ftol':1e-9,'eps':.0005})
        self.calls.append({'nit':int(r.nit),'status':int(r.status),'message':str(r.message),'nfev':int(r.nfev),'njev':int(r.njev),'initial_objective':objective(x0),'final_objective':objective(r.x),'max_power_change_W':float(np.max(np.abs((r.x-x0)*scale)))})
        candidates=[v for v in [x0,r.x] if np.isfinite(v).all() and v.min()>=-1e-8 and v.max()<=1+1e-8 and slew(v).min()>=-1e-5 and thermal(v).min()>=-1e-5]
        if not candidates:return MPCResult(np.full(c.zones,np.nan),np.empty((0,c.zones)),np.inf,0,'INFEASIBLE_CANDIDATES')
        best=min(candidates,key=objective);seq=sequence(best);self.warm=seq.copy()
        return MPCResult(seq[0],seq,objective(best),len(candidates),'OK')

def d10_job(job):
    name,item,frozen_root=job
    import thermono_mpc.controllers as ctl
    p=yaml.safe_load((ROOT/'configs/frozen/protocol_v2.yaml').read_text());c=ProcessConfig(**p['process']).with_grid(*p['evaluation']['grid'])
    dest=OUT/'D10'/f"{item['scenario_id']}_{name}";dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    calls=[];probes=[];controller=[]
    original=ctl.minimize
    def instrument(fun,x0,*a,**kw):
        r=original(fun,x0,*a,**kw)
        calls.append({'call':len(calls),'nit':int(r.nit),'status':int(r.status),'message':str(r.message),'nfev':int(r.nfev),'njev':int(r.njev),'initial_objective':fun(x0),'final_objective':float(r.fun),'max_power_change_W':float(np.max(np.abs(r.x-x0)))})
        if item['scenario_id']=='N03' and len(calls)<=20:
            g=np.array([(fun(x0+np.eye(len(x0))[i])-fun(x0)) for i in range(len(x0))])
            for variant in ['baseline','eps_2W','eps_default','objective_x1e6','variables_unit']:
                opts=dict(kw['options']);f=fun;start=x0.copy();bounds=kw['bounds'];constraints=kw['constraints'];vs=1.
                if variant=='eps_2W':opts['eps']=2.
                if variant=='eps_default':opts.pop('eps')
                if variant=='objective_x1e6':f=lambda v:1e6*fun(v)
                if variant=='variables_unit':
                    vs=2000.;start=x0/vs;f=lambda v:fun(v*2000.)
                    bounds=[(lo/2000.,hi/2000.) for lo,hi in bounds]
                    constraints=[{'type':v['type'],'fun':lambda x,func=v['fun']:func(x*2000.)} for v in constraints];opts['eps']=1./2000.
                q=r if variant=='baseline' else original(f,start,method='SLSQP',bounds=bounds,constraints=constraints,options=opts)
                probes.append({'local_call':len(calls)-1,'selection_call':(len(calls)-1)//2,'variant':variant,'gradient_l2_per_W':float(np.linalg.norm(g)),'gradient_norm_squared':float(g@g),'objective_initial':fun(x0),'objective_final':fun(q.x*vs),'nit':int(q.nit),'nfev':int(q.nfev),'njev':int(q.njev),'status':int(q.status),'message':str(q.message),'max_power_change_W':float(np.max(np.abs(q.x*vs-x0)))})
        return r
    if name=='original':ctl.minimize=instrument;transform=None
    else:
        def transform(base):
            fixed=ScaledSingleStart(base);controller.append(fixed);return fixed
    try:summary,rows=run_episode(c,Scenario(**item),'B1b_ROM_SLSQP',controller_transform=transform,**episode_kwargs(p))
    finally:ctl.minimize=original
    if controller:calls=controller[0].calls
    write_csv(dest/'raw.csv',rows);write_csv(dest/'optimizer_calls.csv',calls);write_csv(dest/'perturbations.csv',probes)
    summary.update({'configuration':name,'optimizer_calls':len(calls),'median_nit':float(np.median([x['nit'] for x in calls])),'maximum_nit':max(x['nit'] for x in calls)})
    if name=='original':
        frozen=Path(frozen_root)/'control_test_v2'/f"{item['scenario_id']}_B1b_ROM_SLSQP_raw.csv"
        write_json(dest/'replay_equality.json',replay_compare(rows,frozen))
    write_json(dest/'summary.json',summary);print('D10',item['scenario_id'],name,summary['rmse_K'],summary['median_nit'],flush=True);return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',choices=['D9','D10'],required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--device',default='cuda');ap.add_argument('--pilot',action='store_true');ap.add_argument('--frozen-evidence',type=Path,default=ROOT/'data/generated/research_inputs');a=ap.parse_args()
    p,cfg,dest=setup(a.id)
    if a.id=='D9':jobs=[(str(q.parent),a.device) for q in sorted((OUT/'D1').glob('*/complete.json'))];fn=d9_job
    else:jobs=[(name,item,str(a.frozen_evidence)) for name in ['original','scaled_single'] for item in p['evaluation']['scenarios']];fn=d10_job
    if a.pilot:jobs=[j for j in jobs if (('N03' in j[0]) if a.id=='D9' else j[1]['scenario_id']=='N03')][:1]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:records=list(pool.map(fn,jobs))
    write_csv(dest/('pilot_summary.csv' if a.pilot else 'summary.csv'),records)
    print(a.id,'COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
