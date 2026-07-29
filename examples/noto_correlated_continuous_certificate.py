from __future__ import annotations
import argparse,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import noto_correlated_validation_postprocess as validation
from ejor_dad.certification import budget_intersecting_grid_cells,continuous_grid_certificate,validate_upper_corner_certificate_instance
from ejor_dad.checkpoint import CheckpointStore,atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y
GRID=np.array([0,.25,.5,.75,1.0]);RHOS=[0,.1,.25]
def evaluate(instance,y):
 r=evaluate_fixed_y(instance,y,epsilon=1e-5,max_iterations=300,enforce_retrofit_budget=False);return {'objective':r.objective,'y':r.y.tolist(),'z':r.z.tolist(),'w':r.w.tolist()}
def main(out,workers):
 root=out/'correlated_facility_full_v1';d=json.loads((out/'run_design.json').read_text());args=validation.args_from(d,out);summary=pd.read_csv(root/'tables/table_noto_correlated_facility.csv');cache=CheckpointStore(root/'certificate_checkpoints');rows=[]
 for rho in RHOS:
  instance=validation.build(rho,args);validate_upper_corner_certificate_instance(instance);cells=budget_intersecting_grid_cells(instance.retrofit_costs,instance.budget_retrofit,GRID);grid_payloads=[]
  for path in (root/'checkpoints').glob(f'*rho{rho:.2f}_*.json'):
   p=json.loads(path.read_text());
   grid_payloads.append(p)
  lookup={tuple(np.round(p['y'],10)):p for p in grid_payloads};values={};pending=[];reused=0;pruned=[]
  for cell in cells:
   key_tuple=tuple(np.round(cell.upper,10));key=f'cert_rho{rho:.2f}_cell{cell.index:04d}'
   if key_tuple in lookup:
    payload=lookup[key_tuple];reused+=1
    if payload.get('status')=='feasible':values[cell.index]=float(payload['objective'])
    else:pruned.append(cell.index)
   elif cache.exists(key):
    payload=cache.load(key)
    if payload.get('status')=='feasible':values[cell.index]=float(payload['objective'])
    else:pruned.append(cell.index)
   else:pending.append((cell,key))
  with ProcessPoolExecutor(max_workers=workers) as pool:
   futures={pool.submit(evaluate,instance,cell.upper):(cell,key) for cell,key in pending}
   for count,f in enumerate(as_completed(futures),1):
    cell,key=futures[f];p=f.result();cache.save(key,p)
    if p.get('status')=='feasible':values[cell.index]=p['objective']
    else:pruned.append(cell.index)
    if count%10==0:atomic_write_text(root/'certificate_status.json',json.dumps({'status':'running','rho':rho,'completed':count,'pending':len(pending),'updated':time.time()},indent=2))
  feasible_cells=tuple(c for c in cells if c.index not in set(pruned));ub=float(summary.loc[np.isclose(summary.rho,rho),'objective'].iloc[0]);cert=continuous_grid_certificate(feasible_cells,[values[c.index] for c in feasible_cells],ub);rows.append({'rho':rho,'continuous_lower_bound':cert.continuous_lower_bound,'grid_upper_bound':cert.grid_upper_bound,'absolute_gap':cert.absolute_gap,'relative_gap_percent':cert.relative_gap_percent,'cell_count':len(cells),'pruned_infeasible_cells':len(pruned),'evaluated_feasible_cells':len(feasible_cells),'reused_grid_upper_corners':reused,'new_or_cached_overbudget_corners':len(cells)-reused,'lower_bound_cell_lower_y_json':json.dumps(cert.lower_bound_cell.lower.tolist()),'lower_bound_cell_upper_y_json':json.dumps(cert.lower_bound_cell.upper.tolist()),'scope':'complete regime-labelled 128-state support; operational Stage 1; monotone upper-corner certificate'})
  pd.DataFrame(rows).to_csv(root/'tables/table_noto_correlated_continuous_certificate.csv',index=False)
 atomic_write_text(root/'certificate_status.json',json.dumps({'status':'completed','rows':len(rows),'updated':time.time()},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();main(Path(a.output_dir),a.workers)

