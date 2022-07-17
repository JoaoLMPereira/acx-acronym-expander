"""
Runs Out Expansion benchmark for the ScienceWISE dataset
"""
import pickle

from Logger import logging
from acronym_expander import RunConfig
from benchmarkers.out_expansion.benchmark import Benchmarker
from helper import getDatasetGeneratedFilesPath, create_configargparser
from string_constants import SCIENCE_WISE_DATASET


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = create_configargparser(
        crossvalidation=True,
        out_expander=True,
        save_and_load=True,
        report_confidences = True,
        results_db_config=True,
    )
    args = parser.parse_args()

    experiment_name = SCIENCE_WISE_DATASET  # pylint: disable=invalid-name
    if args.report_confidences:
        experiment_name += "_confidences"
    
    if args.crossvalidation:
        experiment_name += "_CV"


    run_config = RunConfig(name=experiment_name, save_and_load=args.save_and_load)

    benchmarker = Benchmarker(
        run_config=run_config,
        train_dataset_name=SCIENCE_WISE_DATASET,
        out_expander_name=args.out_expander,
        out_expander_args=args.out_expander_args,
        report_confidences = args.report_confidences
    )

    generatedFilesFolder = getDatasetGeneratedFilesPath(SCIENCE_WISE_DATASET)
    if args.crossvalidation:
        FOLDS_NUM = 5
        foldsFilePath = (
            generatedFilesFolder + str(FOLDS_NUM) + "-cross-validation.pickle"
        )
        folds = pickle.load(open(foldsFilePath, "rb"))

        # Adds index to folds list
        indexedFolds = [(fold[0], fold[1], idx) for idx, fold in enumerate(folds)]
    else:
        trainArticles = pickle.load(
            open(generatedFilesFolder + "train_articles.pickle", "rb")
        )
        testArticles = pickle.load(
            open(generatedFilesFolder + "test_articles.pickle", "rb")
        )
        indexedFolds = [(testArticles, trainArticles)]

    benchmarker.run(
        indexedFolds,
        report_db_config=args.results_database_configuration,
        num_processes=len(indexedFolds),
    )
