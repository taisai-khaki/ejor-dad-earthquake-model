from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import noto_practical_resilience_experiment as practical
import noto_correlated_facility_experiment as correlated
from ejor_dad import generate_regime_failure_states
from ejor_dad.fixed_y import evaluate_fixed_plan,evaluate_fixed_y
from ejor_dad.identification import fixed_y_capacity_ranges
from ejor_dad.tv import capped_tv_profile

def args_from(d,out):return correlated.args_from(d,out)
def build(rho,args):
 base,_=practical.build_instance(rho,args);R=correlated.regimes(base);states=generate_regime_failure_states(base.links,R);critical={s.id for s in states if s.hazard_regime_id in {'normal','north','central'} and len(s.failed_links)<=1};return replace(base,states=states,hazard_regimes=R,critical_service_state_ids=critical,minimum_protected_population=.1*base.protected_population_coefficients.sum(),minimum_zone_service_fraction=.08)
def min_radius(profile,target):
 if target<=float(profile.nominal@profile.values)+1e-10:return 0.0
 for segment in profile.segments:
  if target<=segment.end_value+1e-9:
   if abs(segment.value_slope)<=1e-14:return segment.end_radius
   return segment.start_radius+(target-segment.start_value)/segment.value_slope
 return profile.terminal_radius
def main(out):
 root=out/'correlated_facility_full_v1';summary=pd.read_csv(root/'tables/table_noto_correlated_facility.csv');d=json.loads((out/'run_design.json').read_text());args=args_from(d,out);rho0=summary.iloc[(summary.rho-0).abs().argmin()];y0=np.array(json.loads(rho0.selected_y_json));z0=np.array(json.loads(rho0.selected_z_json));w0=np.array(json.loads(rho0.selected_w_json));adapt=[];prob=[];regime=[];capacity=[]
 for row in summary.itertuples(index=False):
  rho=float(row.rho);instance=build(rho,args);y=np.array(json.loads(row.selected_y_json));selected=evaluate_fixed_y(instance,y,epsilon=1e-8,max_iterations=300);road0=evaluate_fixed_y(instance,y0,epsilon=1e-8,max_iterations=300);complete0=evaluate_fixed_plan(instance,z0,w0,y0);selected_plan=evaluate_fixed_plan(instance,selected.z,selected.w,y)
  road_delta=road0.objective-selected.objective;zw_delta=complete0.objective-road0.objective;all_delta=complete0.objective-selected.objective
  adapt.append({'rho':rho,'road_adaptation_value':road_delta,'zw_given_y0_adaptation_value':zw_delta,'complete_policy_adaptation_value':all_delta,'decomposition_error':all_delta-road_delta-zw_delta,'road_adaptation_percent_of_frozen_nominal_plan':100*road_delta/complete0.objective,'zw_adaptation_percent_of_frozen_nominal_plan':100*zw_delta/complete0.objective,'complete_adaptation_percent_of_frozen_nominal_plan':100*all_delta/complete0.objective,'frozen_nominal_plan_value':complete0.objective,'theta_y0':road0.objective,'selected_value':selected.objective})
  profile=capped_tv_profile(selected.nominal_distribution,selected.state_losses,density_cap=instance.ambiguity_density_cap);minimum_used=min_radius(profile,selected.objective);critical=np.array([state.id in instance.critical_service_state_ids for state in instance.states]);prob.append({'rho':rho,'nominal_sum':selected.nominal_distribution.sum(),'worst_case_sum':selected.worst_case_distribution.sum(),'nominal_sum_error':abs(selected.nominal_distribution.sum()-1),'worst_case_sum_error':abs(selected.worst_case_distribution.sum()-1),'nominal_design_basis_mass':selected.nominal_distribution@critical,'worst_case_design_basis_mass':selected.worst_case_distribution@critical,'actual_tv_movement':.5*np.abs(selected.worst_case_distribution-selected.nominal_distribution).sum(),'minimum_tv_for_value':minimum_used,'radius':rho,'radius_used_percent':100*minimum_used/rho if rho>0 else 0,'nominal_expected_loss':selected.nominal_distribution@selected.state_losses,'robust_loss':selected.objective,'ambiguity_premium':selected.objective-selected.nominal_distribution@selected.state_losses})
  for regime_object in instance.hazard_regimes:
   mask=np.array([state.hazard_regime_id==regime_object.id for state in instance.states]);regime.append({'rho':rho,'regime':regime_object.id,'declared_nominal_weight':regime_object.probability,'computed_nominal_mass':selected.nominal_distribution@mask,'worst_case_mass':selected.worst_case_distribution@mask,'mass_shift':selected.worst_case_distribution@mask-regime_object.probability})
  tie=max(1e-6,1e-5*selected.objective)
  for item in fixed_y_capacity_ranges(instance,y,objective_tolerance=tie,separation_tolerance=1e-6,max_iterations=250):capacity.append({'rho':rho,'center_id':item.center_id,'minimum':item.minimum,'maximum':item.maximum,'width':item.maximum-item.minimum,'tie_tolerance':tie})
 tables=root/'tables';pd.DataFrame(adapt).to_csv(tables/'table_noto_complete_adaptation_values.csv',index=False);pd.DataFrame(prob).to_csv(tables/'table_noto_probability_design_basis_audit.csv',index=False);pd.DataFrame(regime).to_csv(tables/'table_noto_nominal_worst_regime_masses.csv',index=False);pd.DataFrame(capacity).to_csv(tables/'table_noto_correlated_capacity_ranges.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();main(Path(a.output_dir))
