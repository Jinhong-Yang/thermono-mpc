"""D3-D8 post-freeze experiments with separate, inspectable outputs."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import yaml
from scipy.optimize import minimize
from thermono_mpc.controllers import ROMPredictor, MPCConfig, MPCResult, limit_power
from thermono_mpc.dataset import generate_trajectory, generate_dataset, file_sha256
from thermono_mpc.process import ProcessConfig, ThermalPlant
from thermono_mpc.simulation import Scenario, run_episode, make_predictor
from thermono_mpc.physics_loss import finite_volume_residual
from diagnostics import REPO, write_json, write_csv, episode_kwargs
from train import train


class Timed:
    """CUDA events measure stream elapsed time (including host-side idle gaps)."""
    def __init__(self,device): self.device=device
    def __enter__(self):
        if self.device=='cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            self.a,self.b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            self.a.record()
        self.t=time.perf_counter(); return self
    def __exit__(self,*exc):
        if self.device=='cuda': self.b.record(); torch.cuda.synchronize()
        self.record={'wall_s':time.perf_counter()-self.t,'device':self.device,
            'cuda_event_elapsed_s':self.a.elapsed_time(self.b)/1000 if self.device=='cuda' else None,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if self.device=='cuda' else 0,
            'timing_scope':'re-measured post hoc; CUDA event interval includes idle gaps and is not summed kernel execution time'}


def oracle_evaluate(protocol,base,models,trajectories,output,device):
    h=protocol['data']['horizon']; dt=protocol['prediction_test']['control_dt_s']; rows=[]
    for seed,path in trajectories:
        with np.load(path,allow_pickle=False) as z: data={k:z[k] for k in z.files}
        for start in range(len(data['power_W'])-h+1):
            field=data['field_K'][start]; zones=data['zones_K'][start]
            powers=data['power_W'][start:start+h]
            tf=data['field_K'][start+1:start+h+1]; tz=data['zones_K'][start+1:start+h+1]
            for name,modelseed,predictor in models:
                prediction=predictor.predict(field,zones,powers[None],dt)
                pf,pz=prediction.temperature_K[0],prediction.zone_temperature_K[0]
                residual=None
                if name!='ROM':
                    rf,rz=finite_volume_residual(*[torch.as_tensor(x[None],device=device) for x in (pf,pz,field,zones,powers)],dt,base)
                    residual=float((rf.square().mean()+rz.square().mean()).cpu())
                rows.append({'trajectory_seed':seed,'window_start':start,'variant':name,'model_seed':modelseed,
                    'field_rmse_K':float(np.sqrt(np.mean((pf-tf)**2))),
                    'zone_rmse_K':float(np.sqrt(np.mean((pz-tz)**2))),
                    'physics_residual_normalized':residual,'trajectory_sha256':file_sha256(path)})
    write_csv(output/'prediction_windows.csv',rows)
    return rows


def aggregate_predictions(rows):
    output=[]
    for variant in sorted({r['variant'] for r in rows}):
        group=[r for r in rows if r['variant']==variant]
        for metric in ['field_rmse_K','zone_rmse_K','physics_residual_normalized']:
            if all(r[metric] is None for r in group): continue
            values=[]
            for seed in sorted({r['trajectory_seed'] for r in group}):
                # Equal number of windows and training seeds per trajectory.
                values.append(np.mean([r[metric] for r in group if r['trajectory_seed']==seed]))
            values=np.asarray(values); rng=np.random.default_rng(12345)
            boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
            lo,hi=np.percentile(boot,[2.5,97.5])
            output.append({'variant':variant,'metric':metric,'n_trajectories':len(values),
                'mean':float(values.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
    return output


class FiveStartSLSQP:
    """Same ROM/objective/constraints as frozen SLSQP; five feasible starts."""
    def __init__(self,config,settings,seed):
        self.process=config;self.settings=settings;self.predictor=ROMPredictor(config)
        self.warm=None;self.rng=np.random.default_rng(seed);self.calls=[]
    def optimize(self,field_K,zones_K,reference_K,previous_W,dt_s):
        s,c=self.settings,self.process;ref=np.asarray(reference_K)
        indices=np.minimum(np.arange(s.horizon)*s.blocks//s.horizon,s.blocks-1)
        def sequence(flat):return flat.reshape(s.blocks,c.zones)[indices]
        def predict(flat):
            seq=sequence(flat);p=self.predictor.predict(field_K,zones_K,seq[None],dt_s)
            return seq,p.temperature_K[0],p.zone_temperature_K[0]
        def objective(flat):
            seq,f,_=predict(flat);delta=np.diff(np.vstack([previous_W,seq]),axis=0)
            return float(s.tracking_weight*(((f-ref[:,None,None])/s.temperature_scale_K)**2).mean()+
                s.variance_weight*(f.var(axis=(-2,-1))/s.temperature_scale_K**2).mean()+
                s.slew_weight*((delta/s.power_scale_W)**2).mean()+s.energy_weight*seq.sum()*dt_s/s.energy_scale_J)
        def slew(flat):
            delta=np.diff(np.vstack([previous_W,flat.reshape(s.blocks,c.zones)]),axis=0)
            return np.r_[(np.asarray(c.slew)-delta).ravel(),(np.asarray(c.slew)+delta).ravel()]
        def temperature(flat):
            _,f,z=predict(flat);return np.r_[c.solid_limit-f.max(axis=(1,2)),c.zone_limit-z.max(axis=1)]
        def repair(blocks):
            result=np.array(blocks,dtype=float,copy=True);prev=previous_W
            for i in range(s.blocks): result[i]=limit_power(result[i],prev,c);prev=result[i]
            return result.ravel()
        if self.warm is None: guess=np.broadcast_to(np.minimum(previous_W+c.slew,c.pmax),(s.blocks,c.zones))
        else: guess=self.warm[np.minimum(np.arange(s.blocks)*s.horizon//s.blocks+1,s.horizon-1)]
        starts=[repair(guess),repair(np.broadcast_to(c.pmax,(s.blocks,c.zones))),
            repair(np.zeros((s.blocks,c.zones))),repair(np.broadcast_to(np.asarray(c.pmax)/2,(s.blocks,c.zones))),
            repair(self.rng.uniform(0,c.pmax,(s.blocks,c.zones)))]
        candidates=list(starts);statuses=[]
        for start in starts:
            r=minimize(objective,start,method='SLSQP',
                bounds=[(0,c.pmax[i]) for _ in range(s.blocks) for i in range(c.zones)],
                constraints=[{'type':'ineq','fun':slew},{'type':'ineq','fun':temperature}],
                options={'maxiter':80,'ftol':1e-9,'eps':1.,'disp':False})
            candidates.append(r.x);statuses.append({'success':bool(r.success),'status':int(r.status),'nit':int(r.nit)})
        feasible=[(objective(x),sequence(x).copy()) for x in candidates
            if np.all(np.isfinite(x)) and np.all(x>=-1e-5) and np.all(x.reshape(s.blocks,c.zones)<=np.asarray(c.pmax)+1e-5)
            and np.all(slew(x)>=-1e-5) and np.all(temperature(x)>=-1e-5)]
        self.calls.append({'cycle':len(self.calls),'starts':statuses,'feasible_candidates':len(feasible)})
        if not feasible:return MPCResult(np.full(c.zones,np.nan),np.empty((0,c.zones)),np.inf,0,'INFEASIBLE_CANDIDATES')
        value,seq=min(feasible,key=lambda x:x[0]);self.warm=seq
        return MPCResult(seq[0],seq,value,len(feasible),'OK')


def d3(args,p,out):
    cfg=ProcessConfig(**p['process']).with_grid(16,16);plan=p['prediction_test']
    paths=[]
    for seed in plan['trajectory_seeds']:
        path=out/f'trajectory_{seed}.npz'
        trajectory=generate_trajectory(cfg,seed,plan['steps'],plan['control_dt_s'],plan['max_step_s'])
        # Identical future powers/material draws are mandatory for the grid comparison.
        with np.load(args.frozen_evidence/'prediction_test_v2'/path.name,allow_pickle=False) as original:
            assert np.array_equal(trajectory['power_W'],original['power_W'])
            assert np.array_equal(trajectory['material'],original['material'])
        np.savez_compressed(path,**trajectory);paths.append((seed,path))
    models=[('ROM',None,ROMPredictor(cfg))]
    for variant in ['data_only','physics_informed']:
        for seed in p['training']['seeds']:
            models.append((variant,seed,make_predictor(cfg,args.checkpoints/f'{variant}_seed{seed}.json',args.device)))
    with Timed(args.device) as timer:rows=oracle_evaluate(p,cfg,models,paths,out,args.device)
    write_json(out/'timing.json',timer.record);write_csv(out/'prediction_aggregate.csv',aggregate_predictions(rows))


def run_control(args,p,out,scenario,name,checkpoint=None,settings=None,transform=None,run_id=None):
    dest=out/(run_id or scenario['scenario_id']);dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    kw=episode_kwargs(p)
    if settings is not None:kw['mpc_settings']=settings
    cfg=ProcessConfig(**p['process']).with_grid(*p['evaluation']['grid'])
    t=time.perf_counter()
    summary,raw=run_episode(cfg,Scenario(**scenario),name,checkpoint=checkpoint,device=args.device,controller_transform=transform,**kw)
    summary.update(wall_s=time.perf_counter()-t,scope='post-freeze diagnostic',run_id=run_id or scenario['scenario_id'])
    write_csv(dest/'raw.csv',raw);write_json(dest/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['run_id','rmse_K','energy_J','wall_s']}),flush=True)
    return summary


def d5(args,p,out):
    summaries=[]
    for scenario in p['evaluation']['scenarios']:
        if scenario['scenario_id'] not in ['N03','S01','F01']:continue
        for variant,name in [('ROM','B1_ROM_CEM'),('data_only','B2_DATA_NO_MPC')]:
            for candidates in [64,256]:
                for iterations in [3,6]:
                    settings=replace(MPCConfig(**p['evaluation']['mpc']),candidates=candidates,iterations=iterations)
                    run=f"{scenario['scenario_id']}_{variant}_c{candidates}_i{iterations}"
                    checkpoint=None if variant=='ROM' else args.checkpoints/'data_only_seed101.json'
                    r=run_control(args,p,out,scenario,name,checkpoint,settings,run_id=run)
                    summaries.append({**r,'variant':variant,'candidates':candidates,'iterations':iterations})
    write_csv(out/'summary.csv',summaries)


def d6(args,p,out):
    summaries=[]
    for scenario in p['evaluation']['scenarios']:
        cfg=ProcessConfig(**p['process']).with_grid(*p['evaluation']['grid'])
        controller=FiveStartSLSQP(cfg,MPCConfig(**p['evaluation']['mpc']),scenario['seed'])
        r=run_control(args,p,out,scenario,'B1b_ROM_SLSQP',transform=lambda old:controller)
        if controller.calls:write_json(out/scenario['scenario_id']/'optimizer_calls.json',controller.calls)
        summaries.append(r)
    write_csv(out/'summary.csv',summaries)


def d7(args,p,out):
    fields={};rows=[]
    for n in [16,32,64,128]:
        cfg=ProcessConfig(nx=n,ny=n);plant=ThermalPlant(cfg);t=time.perf_counter()
        final=plant.advance(plant.reset(),np.array([800.,1200.,1000.]),300.,2.)
        fields[n]=final.field
        np.savez_compressed(out/f'grid_{n}.npz',field_K=final.field,zones_K=final.zones)
        rows.append({'grid':n,'mean_field_K':float(final.field.mean()),'wall_s':time.perf_counter()-t})
        print('D7',n,rows[-1],flush=True)
    ref=fields[64]
    for r in rows:
        n=r['grid'];f=fields[n]
        if n<64: mapped=np.repeat(np.repeat(f,64//n,axis=0),64//n,axis=1)
        elif n>64:mapped=f.reshape(64,n//64,64,n//64).mean(axis=(1,3))
        else:mapped=f
        r['field_L2_rms_to_64_K']=float(np.sqrt(np.mean((mapped-ref)**2)))
        r['relative_field_L2_to_64']=float(np.linalg.norm(mapped-ref)/np.linalg.norm(ref))
        r['mean_delta_to_64_K']=float(f.mean()-ref.mean())
    write_csv(out/'grid_comparison.csv',rows)


def train_one(args,p,out,weight,epochs):
    t=p['training'];variant='data_only' if weight==0 else 'physics_informed'
    with Timed(args.device) as timer:
        result=train(args.data_manifest,out,variant,101,epochs=epochs,batch_size=t['batch_size'],
            learning_rate=t['learning_rate'],physics_weight=weight,width=t['width'],layers=t['layers'],
            modes=t['modes'],patience=t['patience'],device=args.device)
    write_json(out/'timing.json',timer.record)
    return result


def d4(args,p,out):
    # A separate one-epoch pilot is excluded from the four final training runs.
    pilot=out/'cost_pilot'
    if not (pilot/'timing.json').exists():train_one(args,p,pilot,.01,1)
    pilot_time=json.loads((pilot/'timing.json').read_text())['wall_s']
    estimate=pilot_time*p['training']['epochs']*4*1.5+1800
    estimate_record={'pilot_epoch_wall_s':pilot_time,'training_runs':4,'max_epochs_each':p['training']['epochs'],
        'conservative_estimated_wall_s_on_gpu_host':estimate,'cap_s':14400,'approved':estimate<=14400,
        'method':'pilot wall * 40 epochs * 4 weights * 1.5 margin + 1800 s oracle/control allowance',
        'scope':'single-seed exploratory post-freeze; upper wall-time estimate counts CPU portions conservatively'}
    write_json(out/'cost_estimate.json',estimate_record);print('D4_COST',json.dumps(estimate_record),flush=True)
    if estimate>14400:
        write_json(out/'STOP_REPORT.json',{'reason':'D4 estimate exceeds approved four GPU-hour cap',**estimate_record})
        return
    start=time.perf_counter();rows=[]
    cfg=ProcessConfig(**p['process']).with_grid(*p['prediction_test']['grid'])
    paths=[(s,args.frozen_evidence/'prediction_test_v2'/f'trajectory_{s}.npz') for s in p['prediction_test']['trajectory_seeds']]
    for weight in [0.,.001,.01,.1]:
        dest=out/f'weight_{weight:g}';dest.mkdir(exist_ok=True)
        variant='data_only' if weight==0 else 'physics_informed'
        metadata=dest/f'{variant}_seed101.json'
        if not (dest/'timing.json').exists():
            remaining=14400-(time.perf_counter()-start)
            command=[sys.executable,str(Path(__file__)),'--id','D4','--train-worker','--weight',str(weight),
                '--output',str(args.output),'--frozen-evidence',str(args.frozen_evidence),'--data-manifest',str(args.data_manifest),'--device',args.device]
            with (dest/'train.log').open('w',encoding='utf-8') as f:
                subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,remaining))
        predictor=make_predictor(cfg,metadata,args.device)
        with Timed(args.device) as timer:
            oracle=oracle_evaluate(p,cfg,[(variant,101,predictor)],paths,dest,args.device)
        write_json(dest/'prediction_timing.json',timer.record)
        agg=aggregate_predictions(oracle);write_csv(dest/'prediction_aggregate.csv',agg)
        control=[]
        for scenario in p['evaluation']['scenarios']:
            if scenario['scenario_id'] not in ['N03','S01','F01']:continue
            name='B2_DATA_NO_MPC' if weight==0 else 'B3_PINO_MPC'
            control.append(run_control(args,p,dest,scenario,name,metadata))
        rows.append({'physics_weight':weight,'seed':101,'mean_control_rmse_K':float(np.mean([r['rmse_K'] for r in control])),
            'mean_control_energy_MJ':float(np.mean([r['energy_J']/1e6 for r in control])),
            **{r['metric']:r['mean'] for r in agg},'training_wall_s':json.loads((dest/'timing.json').read_text())['wall_s']})
        write_csv(out/'summary.csv',rows)
        if time.perf_counter()-start>14400:raise RuntimeError('D4 elapsed cap reached; do not start another condition')
    write_json(out/'complete.json',{'wall_s_excluding_pilot':time.perf_counter()-start,'conditions':4,'single_seed':101})


def d8(args,p,out):
    d=p['data'];cfg=ProcessConfig(**p['process'])
    with Timed('cpu') as timer:
        manifest=generate_dataset(out/'regenerated_data',cfg,counts={'train':24,'validation':6,'calibration':6},
            steps=d['steps'],horizon=d['horizon'],control_dt_s=d['control_dt_s'],max_step_s=d['max_step_s'],seed=d['seed'])
    records=[{'task':'36-trajectory data generation',**timer.record,'source':'D8/regenerated_data/manifest.json'}]
    old=json.loads(args.data_manifest.read_text());new=json.loads(manifest.read_text())
    assert [(x['seed'],x['split'],x['sha256']) for x in old['trajectories']]==[(x['seed'],x['split'],x['sha256']) for x in new['trajectories']]
    for task,name in [('one model training','timing.json'),('one model oracle prediction evaluation','prediction_timing.json')]:
        source=args.output/'D4/weight_0.01'/name
        if not source.exists():raise RuntimeError('D8 requires completed D4 weight 0.01 representative timing')
        records.append({'task':task,**json.loads(source.read_text()),'source':str(source.relative_to(args.output))})
    write_csv(out/'offline_costs.csv',records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--id',required=True,choices=['D3','D4','D5','D6','D7','D8'])
    parser.add_argument('--output',type=Path,default=REPO/'outputs/diagnostics_v3')
    parser.add_argument('--checkpoints',type=Path,default=REPO/'artifacts/generated/protocol_v1')
    parser.add_argument('--data-manifest',type=Path,default=REPO/'data/generated/protocol_v1/manifest.json')
    parser.add_argument('--frozen-evidence',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--train-worker',action='store_true');parser.add_argument('--weight',type=float)
    args=parser.parse_args();torch.set_num_threads(4)
    protocol=REPO/'configs/frozen/protocol_v2.yaml';p=yaml.safe_load(protocol.read_text())
    conf=REPO/f'configs/diagnostics/{args.id}.yaml';settings=yaml.safe_load(conf.read_text())
    assert file_sha256(protocol)==settings['base_protocol_sha256']
    out=args.output/args.id;out.mkdir(parents=True,exist_ok=True)
    if args.train_worker:
        train_one(args,p,out/f'weight_{args.weight:g}',args.weight,p['training']['epochs']);return
    write_json(out/'execution.json',{'id':args.id,'scope':'post-freeze diagnostic','started_unix':time.time(),
        'config_sha256':file_sha256(conf),'script_sha256':file_sha256(Path(__file__)),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()),
        'torch_version':torch.__version__,'python_version':sys.version,'device':args.device,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    globals()[args.id.lower()](args,p,out)

if __name__=='__main__':main()
