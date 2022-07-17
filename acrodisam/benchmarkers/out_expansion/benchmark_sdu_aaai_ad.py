"""
Runs Out Expansion benchmark for the ScienceWISE dataset
"""
import pickle

from DataCreators import ArticleDB
from Logger import logging
from acronym_expander import RunConfig
from benchmarkers.out_expansion.benchmark import Benchmarker
from benchmarkers.out_expansion.results_reporter import ResultsReporter
from helper import (
    getDatasetGeneratedFilesPath,
    create_configargparser,
    getArticleDBPath,
    get_raw_article_db_path,
)
from string_constants import (
    SDU_AAAI_AD_DATASET,
    FULL_WIKIPEDIA_DATASET,
    SDU_AAAI_AD_DEDUPE_DATASET,
)


logger = logging.getLogger(__name__)


class SDUAAAIBenchmarker(Benchmarker):  # pylint: disable=too-few-public-methods
    """
    Benchmarker for the SDU AAAI datasets
    When executing the test set there are no expansions available so we skip the test articles
     preprocessing that replaces expansions per acronyms
    """

    def _preprocess_test_article(self, article, actual_expansions):

        if not self.test_dataset_name.endswith("_test"):
            processed_article, new_actual_expansions = super()._preprocess_test_article(
                article, actual_expansions
            )
        else:
            processed_article = article
            new_actual_expansions = actual_expansions

        return processed_article, new_actual_expansions


if __name__ == "__main__":

    parser = create_configargparser(
        crossvalidation=True,
        out_expander=True,
        save_and_load=True,
        report_confidences=True,
        results_db_config=True,
    )

    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="runs test data",
    )

    parser.add_argument(
        "--wikipedia",
        "-wiki",
        action="store_true",
        help="uses wikipedia data for training",
    )

    parser.add_argument(
        "--orig-version",
        "-orig",
        action="store_true",
        help="runs dataset original version instead",
    )

    args = parser.parse_args()

    if args.orig_version:
        dataset_name = SDU_AAAI_AD_DATASET
    else:
        dataset_name = SDU_AAAI_AD_DEDUPE_DATASET

    experiment_name = dataset_name  # pylint: disable=invalid-name
    train_dataset_name = dataset_name
    test_dataset_name = dataset_name
    persistent_articles = None

    if args.wikipedia:
        experiment_name += "_wiki"
        persistent_articles = "SQLITE"
        wikipedia_dataset_name = FULL_WIKIPEDIA_DATASET + "_res-dup-aaai"
        train_dataset_name = [train_dataset_name, wikipedia_dataset_name]

    if args.test:
        experiment_name += "_test"
        test_dataset_name += "_test"

    if args.report_confidences:
        experiment_name += "_confidences"
        
    if args.crossvalidation:
        experiment_name += "_CV"

    run_config = RunConfig(
        name=experiment_name,
        save_and_load=True,
        persistent_articles=persistent_articles,
    )

    benchmarker = SDUAAAIBenchmarker(
        run_config=run_config,
        train_dataset_name=train_dataset_name,
        out_expander_name=args.out_expander,
        out_expander_args=args.out_expander_args,
        results_report=ResultsReporter,
        test_dataset_name=test_dataset_name,
        report_confidences = args.report_confidences
    )

    generatedFilesFolder = getDatasetGeneratedFilesPath(dataset_name)

    train_articles_aaai = pickle.load(
        open(generatedFilesFolder + "train_articles.pickle", "rb")
    )

    dev_articles_aaai = pickle.load(
        open(generatedFilesFolder + "test_articles.pickle", "rb")
    )
    if args.crossvalidation:
        FOLDS_NUM = 5
        foldsFilePath = (
            generatedFilesFolder + str(FOLDS_NUM) + "-cross-validation.pickle"
        )
        folds = pickle.load(open(foldsFilePath, "rb"))

        # Adds index to folds list
        indexedFolds = [(fold[0], fold[1], idx) for idx, fold in enumerate(folds)]
    else:
        if args.test:
            test_articles = list(
                ArticleDB.load(path=get_raw_article_db_path(test_dataset_name)).keys()
            )
            train_articles = train_articles_aaai + dev_articles_aaai
        else:
            train_articles = train_articles_aaai
            test_articles = dev_articles_aaai

        if args.wikipedia:
            wiki_train_articles = list(
                ArticleDB.load(path=getArticleDBPath(wikipedia_dataset_name)).keys()
            )
            train_articles = [train_articles, wiki_train_articles]

        indexedFolds = [(test_articles, train_articles)]

    benchmarker.run(
        indexedFolds,
        report_db_config=args.results_database_configuration,
        num_processes=len(indexedFolds),
    )
