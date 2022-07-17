python acrodisam/benchmarkers/out_expansion/benchmark_science_wise.py --save-and-load --out_expander  cossim --out_expander_args '["concat", "classic_context_vector", [1], "doc2vec", [50, "CBOW", 25, 8]]'
python acrodisam/benchmarkers/out_expansion/benchmark_science_wise.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.1, 0, "concat", ["classic_context_vector", [1], "doc2vec", [50, "CBOW", 25, 8]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_msh.py --save-and-load --out_expander cossim --out_expander_args '["concat", "classic_context_vector", [1], "doc2vec", [25, "CBOW", 200, 2]]'
python acrodisam/benchmarkers/out_expansion/benchmark_msh.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.1, 0, "concat", ["classic_context_vector", [1], "doc2vec", [25, "CBOW", 200, 2]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_cs_wikipedia.py --save-and-load --out_expander cossim --out_expander_args '["concat", "classic_context_vector", [1], "doc2vec", [100, "CBOW", 100, 5]]'
python acrodisam/benchmarkers/out_expansion/benchmark_cs_wikipedia.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.01, 0, "concat", ["classic_context_vector", [1], "doc2vec", [100, "CBOW", 100, 5]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_sdu_aaai_ad.py --save-and-load --out_expander cossim --out_expander_args '["concat", "classic_context_vector", [1], "doc2vec", [50, "CBOW", 200, 5]]'
python acrodisam/benchmarkers/out_expansion/benchmark_sdu_aaai_ad.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.1, 0, "concat", ["classic_context_vector", [1], "doc2vec", [50, "CBOW", 200, 5]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_uad.py --save-and-load --out_expander cossim --out_expander_args '["concat", "classic_context_vector", [1], "doc2vec", [50, "CBOW", 200, 5]]'
python acrodisam/benchmarkers/out_expansion/benchmark_uad.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.1, 0, "concat", ["classic_context_vector", [1], "doc2vec", [50, "CBOW", 200, 5]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_science_wise.py --save-and-load --out_expander  cossim --out_expander_args '["concat", "document_context_vector", [0], "doc2vec", [50, "CBOW", 25, 8]]'
python acrodisam/benchmarkers/out_expansion/benchmark_science_wise.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.1, 0, "concat", ["document_context_vector", [0], "doc2vec", [50, "CBOW", 25, 8]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_msh.py --save-and-load --out_expander cossim --out_expander_args '["concat", "document_context_vector", [0], "doc2vec", [25, "CBOW", 200, 2]]'
python acrodisam/benchmarkers/out_expansion/benchmark_msh.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.1, 0, "concat", ["document_context_vector", [0], "doc2vec", [25, "CBOW", 200, 2]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_cs_wikipedia.py --save-and-load --out_expander cossim --out_expander_args '["concat", "document_context_vector", [0], "doc2vec", [100, "CBOW", 100, 5]]'
python acrodisam/benchmarkers/out_expansion/benchmark_cs_wikipedia.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.01, 0, "concat", ["document_context_vector", [0], "doc2vec", [100, "CBOW", 100, 5]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_sdu_aaai_ad.py --save-and-load --out_expander cossim --out_expander_args '["concat", "document_context_vector", [0], "doc2vec", [50, "CBOW", 200, 5]]'
python acrodisam/benchmarkers/out_expansion/benchmark_sdu_aaai_ad.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.1, 0, "concat", ["document_context_vector", [0], "doc2vec", [50, "CBOW", 200, 5]]]'

python acrodisam/benchmarkers/out_expansion/benchmark_uad.py --save-and-load --out_expander cossim --out_expander_args '["concat", "document_context_vector", [0], "doc2vec", [50, "CBOW", 200, 5]]'
python acrodisam/benchmarkers/out_expansion/benchmark_uad.py --save-and-load --out_expander svm --out_expander_args '["l2", 0.1, 0, "concat", ["document_context_vector", [0], "doc2vec", [50, "CBOW", 200, 5]]]'
