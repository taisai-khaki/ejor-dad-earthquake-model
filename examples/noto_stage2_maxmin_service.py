from __future__ import annotations

import argparse,json,sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import noto_practical_resilience_experiment as practical
from ejor_dad.fixed_y import evaluate_fixed_y

REL_TOLS=[0.001,0.005]; ABS_TOL=1e-5; BISECTION_STEPS=9

def args_from(d,out):
 return SimpleNamespace(mode=d['mode'],density_cap=d['density_cap'],residual_failure_ratio=d['residual_failure_ratio'],failure_delay_reduction=d['failure_delay_reduction'],retrofit_budget_scale=d['retrofit_budget_scale'],time_sensitive_fraction=d['time_sensitive_fraction'],immediate_loss_fraction=d['immediate_loss_fraction'],capacity_throughput_per_bed=d['capacity_throughput_per_bed'],response_threshold_minutes=d.get('response_threshold_minutes'),graded_response=bool(d.get('graded_response',False)),output_dir=str(out),workers=1,force=False)

def evaluate(instance,y):
 try:return evaluate_fixed_y(instance,y,epsilon=1e-5,max_iterations=220)
 except RuntimeError as e:
  if 'infeasible' in str(e).lower():return None
  raise

def main(out):
 design=json.loads((out/'run_design.json').read_text());args=args_from(design,out);source=out/'operational_hard_v1';summary=pd.read_csv(source/'tables'/'table_noto_operational_stage1.csv');files=list((source/'checkpoints').glob('*.json'));rows=[]
 for rho in sorted(summary.rho.unique()):
  benchmark=float(summary.loc[np.isclose(summary.rho,rho),'operational_robust_objective'].iloc[0]);base,_=practical.build_instance(float(rho),args);protected=.10*float(base.protected_population_coefficients.sum())
  payloads=[]
  token=f'rho{rho:.2f}_'
  for path in files:
   if token not in path.name:continue
   p=json.loads(path.read_text());
   if p.get('status')=='feasible':payloads.append(p)
  for rel_tol in REL_TOLS:
   tau=max(ABS_TOL,rel_tol*benchmark);bound=benchmark+tau;candidates=[p for p in payloads if float(p['objective'])<=bound+1e-8];best_records=[]
   for p in candidates:
    y=np.asarray(p['y'],float);low=.08;high=.25;best_result=None
    for _ in range(BISECTION_STEPS):
     mid=(low+high)/2;instance=replace(base,minimum_protected_population=protected,minimum_zone_service_fraction=mid);result=evaluate(instance,y)
     if result is not None and result.objective<=bound+1e-7:low=mid;best_result=result
     else:high=mid
    if best_result is None:
     instance=replace(base,minimum_protected_population=protected,minimum_zone_service_fraction=.08);best_result=evaluate(instance,y)
    best_records.append((low,float(best_result.objective),tuple(y),best_result))
   max_service=max(r[0] for r in best_records);service_ties=[r for r in best_records if r[0]>=max_service-2**-BISECTION_STEPS*.17-1e-10];chosen=min(service_ties,key=lambda r:(r[1],r[2]));service,obj,y_tuple,result=chosen
   rows.append({'rho':rho,'relative_tolerance':rel_tol,'absolute_tolerance':ABS_TOL,'tau':tau,'operational_benchmark':benchmark,'objective_bound':bound,'stage2_service_floor':service,'stage2_objective':obj,'objective_sacrifice':obj-benchmark,'objective_sacrifice_percent':100*(obj/benchmark-1),'selected_y_json':json.dumps(list(y_tuple)),'selected_z_json':json.dumps(result.z.tolist()),'selected_w_json':json.dumps(result.w.tolist()),'stage1_admissible_policy_count':len(candidates),'bisection_steps':BISECTION_STEPS,'status':'grid-optimal over Stage-1-admissible road policies; service floor resolved by bisection'})
  target=out/'operational_stage2_v1'/'tables';target.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(target/'table_noto_stage2_maxmin_service.csv',index=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();main(Path(a.output_dir))
