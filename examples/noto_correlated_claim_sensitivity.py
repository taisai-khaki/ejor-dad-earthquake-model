from __future__ import annotations
import argparse,json,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import noto_correlated_validation_postprocess as validation
import noto_correlated_facility_experiment as corr
import noto_practical_resilience_experiment as practical
from ejor_dad import HazardRegime,generate_regime_failure_states
from ejor_dad.fixed_y import evaluate_fixed_y
A=np.array([.25,1,1,0,0]);B=np.array([0,1,1,0,.25]);RHOS=[0,.1,.25]
def base_regimes(base):return corr.regimes(base)
def normalize_weights(regimes,factor):
 raw=[regimes[0].probability]+[r.probability*factor for r in regimes[1:]];total=sum(raw);return [HazardRegime(r.id,w/total,r.failed_centers,r.link_failure_multipliers) for r,w in zip(regimes,raw)]
def scale_intensity(regimes,factor):return [HazardRegime(r.id,r.probability,r.failed_centers,{k:max(0,1+factor*(v-1)) for k,v in r.link_failure_multipliers.items()}) for r in regimes]
def mild_facilities(regimes):
 patterns={'normal':(),'north':('center_17205',),'central':('center_17461',),'widespread':('center_17202',)};return [HazardRegime(r.id,r.probability,patterns[r.id],r.link_failure_multipliers) for r in regimes]
def settings():
 result=[]
 for f in [.75,1,1.25]:result.append((f'weights_{f:.2f}',{'weights':f}))
 for f in [.75,1,1.25]:result.append((f'intensity_{f:.2f}',{'intensity':f}))
 result += [('facility_mild',{'facility':'mild'}),('facility_base',{})]
 for f in [.05,.10,.20]:result.append((f'residual_{f:.2f}',{'residual':f}))
 for c in [0,.02,.04,.06,.08]:result.append((f'service_{c:.2f}',{'service':c}))
 result += [('design_S1',{'design':'S1'}),('design_S2',{})]
 unique=[];seen=set()
 for name,p in result:
  key=(name,tuple(sorted(p.items())))
  if key not in seen:unique.append((name,p));seen.add(key)
 return unique
def construct(rho,args,params):
 local=type(args)(**(vars(args)|{'residual_failure_ratio':params.get('residual',args.residual_failure_ratio)}));base,_=practical.build_instance(rho,local);R=base_regimes(base)
 if 'weights' in params:R=normalize_weights(R,params['weights'])
 if 'intensity' in params:R=scale_intensity(R,params['intensity'])
 if params.get('facility')=='mild':R=mild_facilities(R)
 states=generate_regime_failure_states(base.links,R)
 if params.get('design')=='S1':critical={s.id for s in states if s.hazard_regime_id=='normal' and len(s.failed_links)<=1}
 else:critical={s.id for s in states if s.hazard_regime_id in {'normal','north','central'} and len(s.failed_links)<=1}
 return replace(base,states=states,hazard_regimes=R,critical_service_state_ids=critical,minimum_protected_population=.1*base.protected_population_coefficients.sum(),minimum_zone_service_fraction=params.get('service',.08))
def solve(instance,y):
 try:return evaluate_fixed_y(instance,y,epsilon=1e-6,max_iterations=300).objective
 except RuntimeError as e:return np.nan
def main(out,workers):
 d=json.loads((out/'run_design.json').read_text());args=validation.args_from(d,out);jobs=[]
 for name,params in settings():
  for rho in RHOS:
   instance=construct(rho,args,params)
   for label,y in [('A',A),('B',B)]:jobs.append((name,params,rho,label,y,instance))
 rows=[]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(solve,i,y):(n,p,r,l) for n,p,r,l,y,i in jobs}
  for f in as_completed(futures):
   n,p,r,l=futures[f];rows.append({'setting':n,'parameters_json':json.dumps(p,sort_keys=True),'rho':r,'policy':l,'objective':f.result()})
 frame=pd.DataFrame(rows);wide=frame.pivot(index=['setting','parameters_json','rho'],columns='policy',values='objective').reset_index();wide['D_A_minus_B']=wide.A-wide.B;wide['preferred']=np.where(wide.D_A_minus_B<0,'A','B');wide['near_tie_0p01_percent']=np.abs(wide.D_A_minus_B)/np.minimum(wide.A,wide.B)<=.0001;wide.to_csv(out/'correlated_facility_full_v1/tables/table_noto_claim_critical_sensitivity.csv',index=False);print(wide.sort_values(['setting','rho']).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();main(Path(a.output_dir),a.workers)

