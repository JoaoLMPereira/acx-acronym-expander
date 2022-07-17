python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander mad_dog --follow-links --out_expander cossim --out_expander_args classic_context_vector
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander mad_dog --follow-links --out_expander cossim --out_expander_args '["doc2vec", [100, "CBOW", 100, 5]]'
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander mad_dog --follow-links --out_expander svm --out_expander_args '["l2", 0.01, 0, "doc2vec", [100, "CBOW", 100, 5]]'
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander mad_dog --out_expander cossim --out_expander_args classic_context_vector
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander mad_dog --out_expander svm --out_expander_args '["l2", 0.01, 0, "doc2vec", [100, "CBOW", 100, 5]]'
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander schwartz_hearst --follow-links --out_expander cossim --out_expander_args classic_context_vector
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander schwartz_hearst --follow-links --out_expander svm --out_expander_args '["l2", 0.01, 0, "doc2vec", [100, "CBOW", 100, 5]]'
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander schwartz_hearst  --out_expander cossim --out_expander_args classic_context_vector
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander schwartz_hearst  --out_expander svm --out_expander_args '["l2", 0.01, 0, "doc2vec", [100, "CBOW", 100, 5]]'
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander mad_dog --follow-links --out_expander cossim --out_expander_args '["tfidf", [[0.5, 1], [1, 3]]]'
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander mad_dog --out_expander cossim --out_expander_args '["tfidf", [[0.5, 1], [1, 3]]]'
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander schwartz_hearst --follow-links --out_expander cossim --out_expander_args '["tfidf", [[0.5, 1], [1, 3]]]'
python3 acrodisam/benchmarkers/end_to_end/benchmark_user_wikipedia.py --in_expander schwartz_hearst --out_expander cossim --out_expander_args '["tfidf", [[0.5, 1], [1, 3]]]'
