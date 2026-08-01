from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
root=Path('data_work/noto/acute_access_graded_v4/correlated_facility_separated_capability_marginal_v2');summary=pd.read_csv(root/'tables/table_noto_correlated_facility.csv');ident=[];top=[]
for rho in summary.rho:
 payloads=[]
 for path in (root/'checkpoints').glob(f'*rho{rho:.2f}_*.json'):
  data=json.loads(path.read_text())
  if data.get('status')=='feasible':payloads.append(data)
 payloads.sort(key=lambda d:(float(d['objective']),tuple(d['y'])));best=float(payloads[0]['objective']);second=float(payloads[1]['objective']);tol=max(1e-8,1e-10*best)
 ident.append({'rho':rho,'best_objective':best,'best_y_json':json.dumps(payloads[0]['y']),'second_best_objective':second,'second_best_y_json':json.dumps(payloads[1]['y']),'absolute_margin':second-best,'margin_percent':100*(second/best-1),'numerical_tie_count':sum(float(d['objective'])<=best+tol for d in payloads),'within_0p01_percent':sum(float(d['objective'])<=best*1.0001 for d in payloads),'within_0p05_percent':sum(float(d['objective'])<=best*1.0005 for d in payloads),'within_0p10_percent':sum(float(d['objective'])<=best*1.001 for d in payloads),'within_0p50_percent':sum(float(d['objective'])<=best*1.005 for d in payloads),'operationally_feasible_count':len(payloads),'operationally_unacceptable_count':996-len(payloads)})
 for rank,data in enumerate(payloads[:10],1):top.append({'rho':rho,'rank':rank,'objective':data['objective'],'gap':float(data['objective'])-best,'gap_percent':100*(float(data['objective'])/best-1),'y_json':json.dumps(data['y']),'z_json':json.dumps(data['z']),'w_json':json.dumps(data['w'])})
ident=pd.DataFrame(ident);ident.to_csv(root/'tables/table_noto_dense_full_grid_identification.csv',index=False);pd.DataFrame(top).to_csv(root/'tables/table_noto_dense_full_grid_top10.csv',index=False)
adapt=pd.read_csv(root/'tables/table_noto_complete_adaptation_values.csv');prob=pd.read_csv(root/'tables/table_noto_probability_design_basis_audit.csv');dense=summary.merge(ident,on='rho').merge(adapt,on='rho').merge(prob,on='rho');dense.to_csv(root/'tables/table_noto_dense_full_grid_radius.csv',index=False)
