from __future__ import annotations
import argparse,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import noto_practical_resilience_experiment as practical
from ejor_dad import HazardRegime,generate_regime_failure_states
from ejor_dad.checkpoint import CheckpointStore,atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y
VERSION='noto-correlated-facility-separated-capability-marginal-v1';RHOS=[0,.025,.05,.075,.1,.125,.15,.2,.25]
def args_from(d,out):return SimpleNamespace(mode=d['mode'],density_cap=d['density_cap'],residual_failure_ratio=d['residual_failure_ratio'],failure_delay_reduction=d['failure_delay_reduction'],retrofit_budget_scale=d['retrofit_budget_scale'],time_sensitive_fraction=d['time_sensitive_fraction'],immediate_loss_fraction=d['immediate_loss_fraction'],capacity_throughput_per_bed=d['capacity_throughput_per_bed'],response_threshold_minutes=d.get('response_threshold_minutes'),graded_response=True,output_dir=str(out),workers=1,force=False)
def regimes(base):
 ids=base.link_ids
 return [HazardRegime('normal',.70,(),{i:.75 for i in ids}),HazardRegime('north',.15,('center_17204','center_17205'),{ids[0]:.9,ids[1]:1.2,ids[2]:1.7,ids[3]:1.4,ids[4]:1.7}),HazardRegime('central',.10,('center_17461','center_17463'),{ids[0]:1.0,ids[1]:1.5,ids[2]:1.5,ids[3]:1.5,ids[4]:1.4}),HazardRegime('widespread',.05,('center_17202','center_17204','center_17205'),{i:1.8 for i in ids})]

def evaluate(instance,y):
 try:
  r=evaluate_fixed_y(instance,y,epsilon=1e-5,max_iterations=240);return {'status':'feasible','objective':r.objective,'y':r.y.tolist(),'z':r.z.tolist(),'w':r.w.tolist(),'iterations':r.iterations}
 except RuntimeError as e:
  if 'infeasible' in str(e).lower():return {'status':'infeasible','y':np.asarray(y).tolist()}
  raise
def main(out,workers):
 d=json.loads((out/'run_design.json').read_text());a=args_from(d,out);target=out/'correlated_facility_separated_capability_marginal_v1';(target/'tables').mkdir(parents=True,exist_ok=True);cache=CheckpointStore(target/'checkpoints');rows=[];levels=[[0,.25,.5,.75,1]]*5
 for rho in RHOS:
  base,_=practical.build_instance(rho,a);R=regimes(base);states=generate_regime_failure_states(base.links,R);critical={s.id for s in states if s.hazard_regime_id in {'normal','north','central'} and len(s.failed_links)<=1};instance=replace(base,states=states,hazard_regimes=R,critical_service_state_ids=critical,minimum_protected_population=.1*base.protected_population_coefficients.sum(),minimum_zone_service_fraction=.08)
  candidates=[]
  for index,values in enumerate(product(*levels),1):
   y=np.asarray(values,float)
   if instance.retrofit_costs@y<=instance.budget_retrofit+1e-9:candidates.append((index,y))
  results=[];pending=[]
  for index,y in candidates:
   key=f'{VERSION}_rho{rho:.2f}_grid{index:04d}'
   if cache.exists(key):
    results.append(cache.load(key))
   else:
    pending.append((index,y,key))
  with ProcessPoolExecutor(max_workers=workers) as pool:
   futures={pool.submit(evaluate,instance,y):(index,key) for index,y,key in pending}
   for count,f in enumerate(as_completed(futures),1):
    index,key=futures[f];r=f.result();r['candidate_index']=index;cache.save(key,r);results.append(r)
    if count%10==0:atomic_write_text(target/'status.json',json.dumps({'status':'running','rho':rho,'completed':count,'pending':len(pending),'updated':time.time()},indent=2))
  feasible=[r for r in results if r['status']=='feasible'];m=min(r['objective'] for r in feasible);tol=max(1e-8,1e-10*m);tied=[r for r in feasible if r['objective']<=m+tol];best=min(tied,key=lambda r:tuple(r['y']));rows.append({'rho':rho,'objective':best['objective'],'selected_y_json':json.dumps(best['y']),'selected_z_json':json.dumps(best['z']),'selected_w_json':json.dumps(best['w']),'feasible_count':len(feasible),'infeasible_count':len(results)-len(feasible),'tie_count':len(tied),'state_count':len(states),'critical_state_count':len(critical),'regime_probabilities_json':json.dumps({r.id:r.probability for r in R}),'grid_scope':'full {0,.25,.5,.75,1}^5 intersect road budget'})
  pd.DataFrame(rows).to_csv(target/'tables'/'table_noto_correlated_facility.csv',index=False)
 atomic_write_text(target/'status.json',json.dumps({'status':'completed','rows':len(rows),'updated':time.time()},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=8);x=p.parse_args();main(Path(x.output_dir),x.workers)


