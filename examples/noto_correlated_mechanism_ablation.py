from __future__ import annotations
import argparse,json,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import noto_correlated_validation_postprocess as validation
import noto_correlated_facility_experiment as correlated
import noto_practical_resilience_experiment as practical
from ejor_dad import HazardRegime,generate_regime_failure_states
from ejor_dad.fixed_y import evaluate_fixed_y
A=np.array([.25,1,1,0,0]);B=np.array([0,1,1,0,.25]);RHOS=[0,.1,.25]
def regime_specs(base):
 shared=correlated.regimes(base);weights={r.id:r.probability for r in shared};avg={link.id:sum(r.probability*r.link_failure_multipliers.get(link.id,1) for r in shared) for link in base.links};facility={r.id:r.failed_centers for r in shared}
 m0=[HazardRegime('independent',1,(),avg)]
 m1=[HazardRegime(r.id,r.probability,(),r.link_failure_multipliers) for r in shared]
 m2=[HazardRegime(r.id,r.probability,r.failed_centers,avg) for r in shared]
 m3=[HazardRegime(f'{rr.id}__{rf.id}',rr.probability*rf.probability,rf.failed_centers,rr.link_failure_multipliers) for rr in shared for rf in shared]
 return {'M0':m0,'M1':m1,'M2':m2,'M3':m3,'M4':shared}
def build(base,regimes,model):
 states=generate_regime_failure_states(base.links,regimes)
 if model=='M0':critical={s.id for s in states if len(s.failed_links)<=1}
 elif model in {'M1','M2','M4'}:critical={s.id for s in states if s.hazard_regime_id in {'normal','north','central'} and len(s.failed_links)<=1}
 else:critical={s.id for s in states if s.hazard_regime_id.split('__')[1] in {'normal','north','central'} and len(s.failed_links)<=1}
 return replace(base,states=states,hazard_regimes=regimes,critical_service_state_ids=critical,minimum_protected_population=.1*base.protected_population_coefficients.sum(),minimum_zone_service_fraction=.08)
def task(instance,y):
 try:r=evaluate_fixed_y(instance,y,epsilon=1e-6,max_iterations=300);return {'status':'feasible','objective':r.objective,'z':r.z.tolist(),'w':r.w.tolist()}
 except RuntimeError as e:return {'status':'infeasible','message':str(e)}
def main(out,workers):
 d=json.loads((out/'run_design.json').read_text());args=validation.args_from(d,out);jobs=[]
 for rho in RHOS:
  base,_=practical.build_instance(rho,args)
  for model,regimes in regime_specs(base).items():
   instance=build(base,regimes,model)
   for label,y in [('A',A),('B',B)]:jobs.append((rho,model,label,y,instance))
 rows=[]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(task,i,y):(rho,m,label,y,len(i.states),len(i.critical_service_state_ids)) for rho,m,label,y,i in jobs}
  for f in as_completed(futures):
   rho,m,label,y,n,ndb=futures[f];r=f.result();rows.append({'rho':rho,'model':m,'policy':label,'y_json':json.dumps(y.tolist()),'state_count':n,'design_basis_count':ndb}|r)
 frame=pd.DataFrame(rows);wide=frame.pivot(index=['rho','model'],columns='policy',values='objective').reset_index();wide['D_A_minus_B']=wide['A']-wide['B'];wide['preferred']=np.where(wide.D_A_minus_B<0,'A','B');wide.to_csv(out/'correlated_facility_separated_capability_marginal_v2/tables/table_noto_mechanism_ablation.csv',index=False);frame.to_csv(out/'correlated_facility_separated_capability_marginal_v2/tables/table_noto_mechanism_ablation_details.csv',index=False);print(wide.sort_values(['model','rho']).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();main(Path(a.output_dir),a.workers)
