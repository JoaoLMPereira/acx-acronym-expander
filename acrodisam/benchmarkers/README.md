# Table of Contents
* [Running the Benchmarks](#running-the-benchmarks)
  * [In-Expansion Benchmark](#in-expansion-benchmark)
  * [Out-Expansion Benchmark](#out-expansion-benchmark)
  * [End-to-end Systems Benchmark](#end-to-end-systems-benchmark)

# Running the Benchmarks
In this section we provide the instructions to run the benchmarks for in-expansion, out-expansion and end-to-end systems described in the paper submitted to VLDB22 entitled 
AcX: system, techniques, and experiments for Acronym eXpansion.

## In-Expansion Benchmark
This benchmark comprises six datasets: Ab3P, BIOADI, MedStract, SH (Schwartz and Hearts), SciAI (SDU@AAAI AI), and User Wikipedia.

### Parse Datasets

First run the parsers for each dataset.

For the Biomedical datasets (Ab3P, BIOADI, MedStract, and SH):

```
$ python3 acrodisam/DatasetParsers/bioc_datasets.py
```


For SciAI:

```
$ python3 acrodisam/DatasetParsers/sdu_aaai_ai.py
```


For User Wikipedia:

```
$ python3 acrodisam/DatasetParsers/user_experiments_wikipedia.py
```

### Run In-expansion Benchmarks for each dataset

Each dataset has its benchmark python script to run:
```
$ python3 acrodisam/benchmarkers/in_expansion/benchmark_ab3p.py
$ python3 acrodisam/benchmarkers/in_expansion/benchmark_bioadi.py
$ python3 acrodisam/benchmarkers/in_expansion/benchmark_medstract.py
$ python3 acrodisam/benchmarkers/in_expansion/benchmark_sh.py
$ python3 acrodisam/benchmarkers/in_expansion/benchmark_sdu_aaai_ai.py
$ python3 acrodisam/benchmarkers/in_expansion/benchmark_user_wikipedia.py
```

To select the in-expander technique set the argument --in_expander followed by either: schwartz_hearst, maddog, scibert_allennlp, or sci_dr.

If scibert_allennlp or sci_dr then an extra argument --in_expander_args can used followed by "{cuda": true}" to enable the use of GPU to train and run the ML models. 

The argument --external_data can be added to use the external data for training the ML models. This is only applicable for scibert_allennlp or sci_dr.

## Out-Expansion Benchmark
This benchmark comprises four datasets: SciWISE, MSH, CSWiki, and SciAD (SDU@AAAI).

### Parse Datasets
First either use the data files we provided for the generated_files folder or run the parsers for each dataset.

For SciWISE:

```
$ python3 acrodisam/DatasetParsers/ScienceWISE.py
```


For MSH:

```
$ python3 acrodisam/DatasetParsers/MSHCorpus.py
```


For CSWiki:

```
$ python3 acrodisam/DatasetParsers/CSWikipediaCorpus.py
$ python3 acrodisam/DatasetParsers/expansion_linkage.py
```


For SciAD:

```
$ python3 acrodisam/DatasetParsers/sdu_aaai_ad.py
```


### Run Out-expansion Benchmarks for each dataset

In order to run the experiments with the tunned hyper-parameters for each dataset, you can execute the following commands:

```
$ sh ./scripts/run_out_exp.sh acrodisam/benchmarkers/out_expansion/benchmark_science_wise.py scripts/sciencewise_best_exp_param.csv
$ sh ./scripts/run_out_exp.sh acrodisam/benchmarkers/out_expansion/benchmark_msh.py scripts/msh_best_exp_param.csv
$ sh ./scripts/run_out_exp.sh acrodisam/benchmarkers/out_expansion/benchmark_cs_wikipedia.py scripts/cswikipedia_best_exp_param.csv
$ sh ./scripts/run_out_exp.sh acrodisam/benchmarkers/out_expansion/benchmark_sdu_aaai_ad.py scripts/sdu_best_exp_param.csv 
```

To execute the experiments with out-expanders with fixed parameters (i.e., baselines and related work) for each dataset, you can execute the following commands:
```
$ sh ./scripts/run_out_exp.sh acrodisam/benchmarkers/out_expansion/benchmark_science_wise.py scripts/out_expanders_fixed_exp_param.csv
$ sh ./scripts/run_out_exp.sh acrodisam/benchmarkers/out_expansion/benchmark_msh.py scripts/out_expanders_fixed_exp_param.csv
$ sh ./scripts/run_out_exp.sh acrodisam/benchmarkers/out_expansion/benchmark_cs_wikipedia.py scripts/out_expanders_fixed_exp_param.csv
$ sh ./scripts/run_out_exp.sh acrodisam/benchmarkers/out_expansion/benchmark_sdu_aaai_ad.py scripts/out_expanders_fixed_exp_param.csv
```

## End-to-end Systems Benchmark

First either use the data files we provided for the generated_files folder or run the parsers:

```
$ python3 acrodisam/DatasetParsers/FullWikipedia.py
$ python3 acrodisam/DatasetParsers/user_experiments_wikipedia.py
```


To run the system benchmark for the AcX pipelines please run the following script:

```
$ sh ./scripts/run_end-to-end.sh
```

To run the MadDog system with original models make sure to download the original models from https://archive.org/details/MadDog-models and place them in folder data/PreTrainedModels/MadDog .

Then, run the following command:

```
$ python3  acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander maddog --out_expander maddog --out_expander_args '{use_original_models: "True"}'
```
