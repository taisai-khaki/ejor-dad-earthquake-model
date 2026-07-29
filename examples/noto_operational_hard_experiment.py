from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noto_practical_resilience_experiment as practical
from ejor_dad.checkpoint import CheckpointStore, atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y

VERSION = "noto-operational-hard-v1"
GRID = np.array([0,.25,.5,.75,1.0])


def payload(result):
 return {"objective":result.objective,"lower_bound":result.lower_bound,"z":result.z.tolist(),"w":result.w.tolist(),"y":result.y.tolist(),"iterations":result.iterations}

def evaluate(instance,y):
 try: return {"status":"feasible"}|payload(evaluate_fixed_y(instance,y,epsilon=1e-5,max_iterations=200))
 except RuntimeError as e:
  if "infeasible" in str(e).lower(): return {"status":"infeasible","message":str(e),"y":np.asarray(y).tolist()}
  raise

def args_from_design(d,out):
 return SimpleNamespace(mode=d['mode'],density_cap=d['density_cap'],residual_failure_ratio=d['residual_failure_ratio'],failure_delay_reduction=d['failure_delay_reduction'],retrofit_budget_scale=d['retrofit_budget_scale'],time_sensitive_fraction=d['time_sensitive_fraction'],immediate_loss_fraction=d['immediate_loss_fraction'],capacity_throughput_per_bed=d['capacity_throughput_per_bed'],response_threshold_minutes=d.get('response_threshold_minutes'),graded_response=bool(d.get('graded_response',False)),output_dir=str(out),workers=1,force=False)

def main(out,workers):
 design=json.loads((out/'run_design.json').read_text(encoding='utf-8')); args=args_from_design(design,out)
 target=out/'operational_hard_v1'; (target/'checkpoints').mkdir(parents=True,exist_ok=True); (target/'tables').mkdir(exist_ok=True)
 cache=CheckpointStore(target/'checkpoints'); rows=[]
 for rho in sorted(design['rho_values']):
  base,_=practical.build_instance(float(rho),args)
  protected=0.10*float(base.protected_population_coefficients.sum())
  instance=replace(base,minimum_protected_population=protected,minimum_zone_service_fraction=.08)
  candidates=[]
  for index,values in enumerate(product(GRID,repeat=len(instance.links)),start=1):
   y=np.asarray(values,float)
   if instance.retrofit_costs@y <= instance.budget_retrofit+1e-9: candidates.append((index,y))
  results=[]; pending=[]
  for index,y in candidates:
   key=f"{VERSION}_rho{rho:.2f}_grid{index:04d}"
   if cache.exists(key): results.append(cache.load(key))
   else: pending.append((index,y,key))
  with ProcessPoolExecutor(max_workers=workers) as pool:
   futures={pool.submit(evaluate,instance,y):(key,index) for index,y,key in pending}
   for count,future in enumerate(as_completed(futures),start=1):
    key,index=futures[future]; result=future.result(); result['candidate_index']=index; cache.save(key,result); results.append(result)
    if count%25==0: atomic_write_text(target/'status.json',json.dumps({"status":"running","rho":rho,"completed_new":count,"pending":len(pending),"updated":time.time()},indent=2))
  feasible=[r for r in results if r['status']=='feasible']
  if not feasible: raise RuntimeError(f'No operationally feasible policy at rho={rho}')
  minimum=min(float(r['objective']) for r in feasible); tol=max(1e-8,1e-10*minimum)
  tied=[r for r in feasible if float(r['objective'])<=minimum+tol]
  best=min(tied,key=lambda r:tuple(r['y']))
  rows.append({"rho":rho,"operational_robust_objective":best['objective'],"selected_y_json":json.dumps(best['y']),"selected_z_json":json.dumps(best['z']),"selected_w_json":json.dumps(best['w']),"feasible_policy_count":len(feasible),"infeasible_operational_count":len(results)-len(feasible),"tie_count":len(tied),"minimum_protected_population":protected,"minimum_zone_timely_service_fraction":.08,"critical_state_set":"all 32 complete no-tail states","solution_scope":"exact fixed-y oracle over complete 5-level road grid"})
  pd.DataFrame(rows).to_csv(target/'tables'/'table_noto_operational_stage1.csv',index=False)
 atomic_write_text(target/'status.json',json.dumps({"status":"completed","rows":len(rows),"updated":time.time()},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1)); a=p.parse_args(); main(Path(a.output_dir),a.workers)
