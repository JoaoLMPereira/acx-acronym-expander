#!/bin/bash
python acrodisam/benchmarkers/out_expansion/benchmark_uad.py --out_expander sci_dr --out_expander_args '["base",32]'
python acrodisam/benchmarkers/out_expansion/benchmark_uad.py --out_expander sci_dr --out_expander_args '["both",32]'

python acrodisam/benchmarkers/out_expansion/benchmark_science_wise.py --out_expander sci_dr --out_expander_args '["base",32]'
python acrodisam/benchmarkers/out_expansion/benchmark_science_wise.py --out_expander sci_dr --out_expander_args '["both",32]'

python acrodisam/benchmarkers/out_expansion/benchmark_msh.py --out_expander sci_dr --out_expander_args '["base",32]'
python acrodisam/benchmarkers/out_expansion/benchmark_msh.py --out_expander sci_dr --out_expander_args '["both",32]'

python acrodisam/benchmarkers/out_expansion/benchmark_cs_wikipedia.py --out_expander sci_dr --out_expander_args '["base",32]'
python acrodisam/benchmarkers/out_expansion/benchmark_cs_wikipedia.py --out_expander sci_dr --out_expander_args '["both",32]'

python acrodisam/benchmarkers/out_expansion/benchmark_sdu_aaai_ad.py --out_expander sci_dr --out_expander_args '["base",32]'
python acrodisam/benchmarkers/out_expansion/benchmark_sdu_aaai_ad.py --out_expander sci_dr --out_expander_args '["both",32]'
