# Reviewer Response: Full 720-Instance NILS--ALNS Comparison

We replaced the previous 24-instance matched subset comparison with a full matched comparison over all 720 instances in the experimental design. The ALNS benchmark was run for every instance and matched one-to-one with the corresponding NILS result from the full computational study.

## Overall result

- Matched instances: 720.
- Feasible/reportable paired instances: 720.
- Feasibility: NILS 100.0%, ALNS 100.0%.
- Wins/losses/ties for NILS vs ALNS: 481/234/5.
- Mean objective: NILS 48038.472, ALNS 50663.365.
- Median objective: NILS 2836.167, ALNS 6831.299.
- Mean/median NILS improvement over ALNS: 15.093% / 17.364%.
- Mean CPU: NILS 11.777s, ALNS 9.938s.
- Median CPU: NILS 4.178s, ALNS 10.641s.
- Wilcoxon signed-rank test: statistic=80259.000, p=5.703e-18, rank-biserial=-0.373.

## Size-stratified interpretation

- n=15: NILS wins/losses/ties = 138/87/5; median improvement = 1.324%; Wilcoxon p=7.635e-09.
- n=50: NILS wins/losses/ties = 207/3/0; median improvement = 56.382%; Wilcoxon p=7.290e-35.
- n=75: NILS wins/losses/ties = 65/25/0; median improvement = 38.399%; Wilcoxon p=3.376e-03.
- n=100: NILS wins/losses/ties = 71/119/0; median improvement = -26.874%; Wilcoxon p=4.981e-03.

The only size class where ALNS has a lower median objective is n=100; NILS remains better overall across the full matched design and especially strong for n=50 and n=75 instances.