"""
Runs Out Expansion benchmark for the CS Wikipedia dataset
"""
import pickle

from Logger import logging
from acronym_expander import RunConfig
from benchmarkers.out_expansion.benchmark import Benchmarker
from helper import getDatasetGeneratedFilesPath, create_configargparser,\
    flatten_to_str_list
from string_constants import CS_WIKIPEDIA_DATASET


logger = logging.getLogger(__name__)

dataset_name = CS_WIKIPEDIA_DATASET + "_res-dup"

# sets defaults
# tfidfArgs = ['1000-x-x ', '1-1']
# ldaArgs = ['100', 'log(nub_distinct_words)+1']
# doc2vecArgs= ['50', 'CBOW', '200', '8']

# newLocalityContextVectorArgs = ["1","0.8"]
# newLocalityContextVectorSmallArgs = ["1"]
# newLocalityContextVectorSmallParaArgs = newLocalityContextVectorArgs + ["3"]

if __name__ == "__main__":

    parser = create_configargparser(
        crossvalidation=True, out_expander=True, save_and_load=True,
        results_db_config=True, report_confidences=True,
    )
    args = parser.parse_args()

    experiment_name = dataset_name  # pylint: disable=invalid-name
    if args.report_confidences:
        experiment_name += "_confidences"
    
    if args.crossvalidation:
        experiment_name += "_CV"

    run_config = RunConfig(
        name=experiment_name,
        save_and_load=args.save_and_load,
        persistent_articles="SQLITE",
    )

    benchmarker = Benchmarker(
        run_config=run_config,
        train_dataset_name=dataset_name,
        out_expander_name=args.out_expander,
        out_expander_args=args.out_expander_args,
        report_confidences = args.report_confidences
    )

    generatedFilesFolder = getDatasetGeneratedFilesPath(dataset_name)
    newSuffix = "_new"
    if args.crossvalidation:
        FOLDS_NUM = 5
        foldsFilePath = (
            generatedFilesFolder
            + str(FOLDS_NUM)
            + "-cross-validation"
            + newSuffix
            + ".pickle"
        )
        folds = pickle.load(open(foldsFilePath, "rb"))

        # Adds index to folds list
        indexedFolds = [(fold[0], fold[1], idx) for idx, fold in enumerate(folds)]
    else:
        trainArticles = pickle.load(
            open(generatedFilesFolder + "train_articles" + newSuffix + ".pickle", "rb")
        )
        testArticles = pickle.load(
            open(generatedFilesFolder + "test_articles" + newSuffix + ".pickle", "rb")
        )
        indexedFolds = [(testArticles, trainArticles)]

    
    out_expander_args_values = flatten_to_str_list(args.out_expander_args)

    if args.crossvalidation and 'LDA' in out_expander_args_values:
        num_processes = len(indexedFolds)
    elif 'tfidf' in out_expander_args_values:
        idx = out_expander_args_values.index('tfidf')
        if len(out_expander_args_values) > idx + 4 and out_expander_args_values[idx + 3] == '1' and out_expander_args_values[idx + 4] == '3':
            num_processes = 3
        else:
            num_processes = len(indexedFolds)
    #elif args.crossvalidation and len(argv) > 3 and argv[3] == '3-3':
    #    num_processes = 5
    #elif args.crossvalidation and (len(argv) > 0 and argv[0] == 'NGramsContextVector' 
    #             or len(argv) > 1 and argv[1] == 'NGramsContextVector'):
    #    num_processes = 2
    else:
        num_processes = len(indexedFolds)
    

    benchmarker.run(indexedFolds, report_db_config=args.results_database_configuration, num_processes=num_processes)
