# eval
Loading dataset from datasets/sample...
Loaded 10 passages, 12 questions
Loading config from configs/problem4.yaml...
Connecting to team server at http://localhost:8001...
Health check passed.
Starting evaluation: 3 runs, 12 questions each
Running 2 warmup requests...
Warmup complete.
Run 1/3...
Run 2/3...
Run 3/3...
Computing medians...
Computing metrics...

======================================================================
  RESULTS: good
======================================================================
  Final Score: 95.00 / 100
  Success Rate: 100.0%
======================================================================
  Metric                           Value    Score   Weight    Dir
  ------------------------------------------------------------
  recall_at_5                      0.800   0.750    0.20 higher
  mrr                              0.756   1.000    0.15 higher
  ttfa_p50_ms                      173.4   1.000    0.20  lower
  ttfa_p95_ms                      190.2   1.000    0.10  lower
  stall_ms_mean                      0.0   1.000    0.15  lower
  overlap_ratio_mean               0.931   1.000    0.10 higher
  total_p50_ms                    2594.6   1.000    0.10  lower
  recall_at_1                      0.700     N/A      N/A higher
  recall_at_3                      0.800     N/A      N/A higher
  retrieval_p50_ms                  22.5     N/A      N/A  lower
  stall_count_mean                   0.0     N/A      N/A  lower
======================================================================

Results written to results/good.json