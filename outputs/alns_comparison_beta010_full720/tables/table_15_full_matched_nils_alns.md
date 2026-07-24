# Full 720-Instance NILS--ALNS Comparison

Lower objective values are better. Positive improvement means NILS improves over ALNS.
Wilcoxon tests are paired on instances where both methods are feasible/reportable.
Negative rank-biserial effect sizes favor NILS because the test is applied to NILS minus ALNS objectives.

| size | matched_instances | paired_feasible_instances | nils_feasible_pct | alns_feasible_pct | nils_wins | alns_wins | ties | mean_nils_objective | median_nils_objective | mean_alns_objective | median_alns_objective | mean_nils_improvement_vs_alns_pct | median_nils_improvement_vs_alns_pct | mean_nils_cpu_s | median_nils_cpu_s | mean_alns_cpu_s | median_alns_cpu_s | wilcoxon_test | wilcoxon_statistic | wilcoxon_p_value | wilcoxon_effect_size | wilcoxon_effect_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All | 720 | 720 | 100 | 100 | 481 | 234 | 5 | 48038.5 | 2836.17 | 50663.4 | 6831.3 | 15.0929 | 17.364 | 11.777 | 4.17796 | 9.93764 | 10.6408 | wilcoxon | 80259 | 5.70256e-18 | -0.372903 | rank_biserial |
| 15 | 230 | 230 | 100 | 100 | 138 | 87 | 5 | 212.973 | 116.223 | 870.848 | 125.528 | 17.3038 | 1.32363 | 0.296308 | 0.267208 | 6.55155 | 6.43634 | wilcoxon | 7066 | 7.63523e-09 | -0.444169 | rank_biserial |
| 50 | 210 | 210 | 100 | 100 | 207 | 3 | 0 | 14036.3 | 3992.99 | 34504.6 | 22518.1 | 56.9846 | 56.3824 | 3.79167 | 3.68687 | 11.3827 | 10.5697 | wilcoxon | 218 | 7.28954e-35 | -0.98032 | rank_biserial |
| 75 | 90 | 90 | 100 | 100 | 65 | 25 | 0 | 39808.1 | 11982.5 | 43577.6 | 20003.2 | 18.4522 | 38.3991 | 10.7087 | 8.87294 | 10.843 | 10.7704 | wilcoxon | 1319 | 0.00337593 | -0.3558 | rank_biserial |
| 100 | 190 | 190 | 100 | 100 | 71 | 119 | 0 | 147412 | 109235 | 132155 | 99029 | -35.4759 | -26.8737 | 35.0065 | 31.0545 | 12.0106 | 11.8668 | wilcoxon | 6941 | 0.00498123 | 0.234941 | rank_biserial |
