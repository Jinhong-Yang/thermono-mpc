"""Post-freeze diagnostics; never overwrites confirmatory evaluation artifacts."""
from __future__ import annotations
import argparse, csv, json, subprocess, time
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import yaml
from thermono_mpc.controllers import CEMMPC, MPCConfig, ROMPredictor
from thermono_mpc.dataset import file_sha256
from thermono_mpc.process import ProcessConfig, ThermalPlant
from thermono_mpc.simulation import Scenario, run_episode

REPO = Path(__file__).resolve().parents[1]

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def context(args):
    protocol_path=REPO/'configs/frozen/protocol_v2.yaml'
    protocol=yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
    settings=yaml.safe_load((REPO/f'configs/diagnostics/{args.id}.yaml').read_text(encoding='utf-8'))
    # Check against the receipt supplied with the frozen archive, not an altered protocol.
    if settings.get('base_protocol_sha256') and file_sha256(protocol_path)!=settings['base_protocol_sha256']:
        raise ValueError('Base protocol hash mismatch')
    output=args.output/args.id
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'execution.json',{'scope':'post-freeze diagnostic', 'diagnostic':args.id,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()),
        'script_sha256':file_sha256(Path(__file__)), 'config_sha256':file_sha256(REPO/f'configs/diagnostics/{args.id}.yaml'),
        'base_protocol_sha256':file_sha256(protocol_path),'device':args.device,'pilot_cycles':args.pilot_cycles,
        'started_unix':time.time()})
    return protocol,settings,output

def episode_kwargs(protocol):
    e=protocol['evaluation']
    return dict(steps=e['episode_steps'],dt_s=e['control_dt_s'],max_step_s=e['max_step_s'],
        horizon=e['mpc']['horizon'],target_K=e['reference_target_K'],ramp_s=e['reference_ramp_s'],
        observation=e['observation'],mpc_settings=MPCConfig(**e['mpc']),deadline_s=e['runtime_deadline_s'],pid_gains=tuple(e['pid_gains']))

class PlanTrace:
    def __init__(self, dt_s, max_step_s):
        self.dt_s,self.max_step_s=dt_s,max_step_s
        self.iterations=[]; self.cycles=[]; self.arrays={}; self.metrics=[]
    def capture_candidates(self,event): self.iterations.append(event)
    def configure(self,mpc):
        mpc.diagnostic_hook=self.capture_candidates
        return mpc
    def record(self,event):
        cycle=event['cycle']; result=event['result']
        if result is None or result.status!='OK':
            self.iterations=[]
            raise RuntimeError(f'D1 missing optimized plan at cycle {cycle}; preserve as an explicit incomplete run')
        best=min((x for x in self.iterations if len(x['elite_indices'])),key=lambda x:x['cost'][x['elite_indices'][0]])
        predicted_f,predicted_z=best['iteration_best_field_K'],best['iteration_best_zone_K']
        predictor=event['controller'].predictor
        aligned=event['state_after_held']
        oracle=predictor.predict(aligned.field,aligned.zones,result.sequence_W[None],self.dt_s)
        plant=ThermalPlant(event['actual_config']); current=aligned.copy()
        fields=[]; zones=[]
        for power in result.sequence_W:
            current=plant.advance(current,power,self.dt_s,self.max_step_s)
            fields.append(current.field); zones.append(current.zones)
        actual_f,actual_z=np.stack(fields),np.stack(zones)
        prefix=f'cycle_{cycle:03d}_'
        payload={
            'plan_W':result.sequence_W,'reference_K':event['reference_K'],
            'predicted_field_K':predicted_f,'predicted_zone_K':predicted_z,
            'oracle_field_K':oracle.temperature_K[0],'oracle_zone_K':oracle.zone_temperature_K[0],
            'realized_field_K':actual_f,'realized_zone_K':actual_z,
            'state_before_field_K':event['state_before'].field,'state_before_zone_K':event['state_before'].zones,
            'aligned_true_field_K':aligned.field,'aligned_true_zone_K':aligned.zones,
            'estimated_field_K':event['estimate'].field_K,'estimated_zone_K':event['estimate'].zones_K,
            'projected_field_K':event['projected_field_K'],'projected_zone_K':event['projected_zone_K'],
            'held_power_W':event['held_power_W'],
            'candidate_plans_W':np.stack([x['sequences_W'] for x in self.iterations]),
            'candidate_cost':np.stack([x['cost'] for x in self.iterations]),
            'elite_indices':np.stack([x['elite_indices'] for x in self.iterations]),
        }
        self.arrays.update({prefix+k:np.array(v,copy=True) for k,v in payload.items()})
        error=predicted_f-actual_f
        oracle_error=oracle.temperature_K[0]-actual_f
        row={'cycle':cycle,'selection_time_s':event['state_before'].time_s,
            'horizon_end_time_s':aligned.time_s+len(result.sequence_W)*self.dt_s,
            'actual_current_mean_K':float(event['state_before'].field.mean()),
            'reference_end_K':float(event['reference_K'][-1]),
            'predicted_end_mean_K':float(predicted_f[-1].mean()),'realized_end_mean_K':float(actual_f[-1].mean()),
            'end_mean_bias_K':float(error[-1].mean()),'end_field_rmse_K':float(np.sqrt(np.mean(error[-1]**2))),
            'horizon_field_rmse_K':float(np.sqrt(np.mean(error**2))),
            'end_zone_bias_K':float((predicted_z[-1]-actual_z[-1]).mean()),
            'end_zone_rmse_K':float(np.sqrt(np.mean((predicted_z[-1]-actual_z[-1])**2))),
            'oracle_end_mean_bias_K':float(oracle_error[-1].mean()),
            'oracle_end_field_rmse_K':float(np.sqrt(np.mean(oracle_error[-1]**2))),
            'projected_state_rmse_K':float(np.sqrt(np.mean((event['projected_field_K']-aligned.field)**2))),
            'planned_energy_J':float(result.sequence_W.sum()*self.dt_s),
            'selected_first_power_total_W':float(result.power_W.sum())}
        for z in range(actual_z.shape[1]): row[f'end_zone_{z}_bias_K']=float(predicted_z[-1,z]-actual_z[-1,z])
        self.metrics.append(row); self.iterations=[]

def replay_compare(rows, frozen_path):
    with frozen_path.open(encoding='utf-8') as f: original=list(csv.DictReader(f))
    out={'cycles_compared':len(rows)}
    for key in ['mean_field_K','field_rmse_K','applied_power_W','selected_next_power_W']:
        a=np.array([r[key] for r in rows],dtype=float)
        b=np.array([json.loads(r[key]) if key.endswith('_W') else float(r[key]) for r in original[:len(rows)]])
        out[key+'_max_abs_diff']=float(np.max(np.abs(a-b)))
    return out

def d1(args,protocol,settings,output):
    e=protocol['evaluation']; config=ProcessConfig(**protocol['process']).with_grid(*e['grid'])
    jobs=[('ROM',None,'B1_ROM_CEM')]+[(v,s,n) for v,n in [('data_only','B2_DATA_NO_MPC'),('physics_informed','B3_PINO_MPC')] for s in settings['model_seeds']]
    summaries=[]; replay=[]
    for item in e['scenarios']:
        if item['scenario_id'] not in settings['representative_scenarios']: continue
        if args.scenario and item['scenario_id']!=args.scenario: continue
        for variant,seed,name in jobs:
            if args.variant and variant!=args.variant: continue
            if args.seed and seed!=args.seed: continue
            run=f"{item['scenario_id']}_{name}"+(f'_seed{seed}' if seed else '')
            dest=output/run; dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():
                record=json.loads((dest/'complete.json').read_text()); summaries.append(record)
                replay.append({'run_id':run,**json.loads((dest/'replay_comparison.json').read_text())})
                continue
            print('D1',run,flush=True); tic=time.perf_counter()
            kwargs=episode_kwargs(protocol)
            if args.pilot_cycles: kwargs['steps']=args.pilot_cycles
            checkpoint=None if seed is None else args.checkpoints/f'{variant}_seed{seed}.json'
            if checkpoint:
                meta=json.loads(checkpoint.read_text()); assert file_sha256(checkpoint.parent/meta['weights'])==meta['weights_sha256']
            trace=PlanTrace(kwargs['dt_s'],kwargs['max_step_s'])
            summary,rows=run_episode(config,Scenario(**item),name,checkpoint=checkpoint,device=args.device,
                controller_transform=trace.configure,diagnostic_hook=trace.record,**kwargs)
            write_csv(dest/'closed_loop.csv',rows); write_csv(dest/'selected_plans.csv',trace.metrics)
            np.savez_compressed(dest/'full_trace.npz',**trace.arrays)
            comparison=replay_compare(rows,args.frozen_evidence/'control_test_v2'/f'{run}_raw.csv')
            write_json(dest/'replay_comparison.json',comparison); replay.append({'run_id':run,**comparison})
            record={'run_id':run,'variant':variant,'model_seed':seed,'scenario_id':item['scenario_id'],
                'cycles':len(rows),'wall_s':time.perf_counter()-tic,'closed_loop_rmse_K':summary['rmse_K'],
                'closed_loop_energy_J':summary['energy_J'],
                **{k:float(np.mean([r[k] for r in trace.metrics])) for k in trace.metrics[0] if k not in ['cycle','selection_time_s','horizon_end_time_s']},
                'full_trace_sha256':file_sha256(dest/'full_trace.npz')}
            write_json(dest/'complete.json',record); summaries.append(record)
            print(json.dumps({k:record[k] for k in ['run_id','wall_s','end_mean_bias_K','oracle_end_mean_bias_K']}),flush=True)
            if args.pilot_cycles:
                write_csv(output/'pilot_summary.csv',summaries); return
    if not args.scenario and not args.variant:
        write_csv(output/'episode_summary.csv',summaries); write_csv(output/'replay_comparison.csv',replay)

def parallel_d1(args, settings):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logdir=args.output/'D1/logs'; logdir.mkdir(parents=True,exist_ok=True)
    jobs=[(scenario,v,s) for scenario in settings['representative_scenarios']
          for v,s in [('ROM',None)]+[(v,s) for v in ['data_only','physics_informed'] for s in settings['model_seeds']]]
    def launch(job):
        scenario,variant,seed=job; name=f'{scenario}_{variant}_{seed}'
        command=[sys.executable,str(Path(__file__)), '--id','D1','--frozen-evidence',str(args.frozen_evidence),
            '--output',str(args.output),'--checkpoints',str(args.checkpoints),'--device',args.device,
            '--scenario',scenario,'--variant',variant]
        if seed: command+=['--seed',str(seed)]
        print('START',name,flush=True)
        with (logdir/f'{name}.log').open('w',encoding='utf-8') as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=REPO)
        return name
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(launch,j) for j in jobs]
        for future in as_completed(futures): print('COMPLETE',future.result(),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--id',required=True,choices=['D1'])
    parser.add_argument('--output',type=Path,default=REPO/'outputs/diagnostics_v3')
    parser.add_argument('--checkpoints',type=Path,default=REPO/'artifacts/generated/protocol_v1')
    parser.add_argument('--frozen-evidence',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--pilot-cycles',type=int,default=0)
    parser.add_argument('--scenario')
    parser.add_argument('--variant')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.pilot_cycles and args.output==REPO/'outputs/diagnostics_v3':
        parser.error('Pilot must use a separate output directory')
    import torch
    torch.set_num_threads(4)
    protocol,settings,output=context(args)
    if args.workers>1:
        parallel_d1(args,settings)
    d1(args,protocol,settings,output)

if __name__=='__main__': main()
