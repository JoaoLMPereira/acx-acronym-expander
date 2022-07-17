import pickle
from helper import create_configargparser, getDatasetGeneratedFilesPath
from Logger import logging
from benchmarkers.in_expansion.results_reporter import ResultsReporter
from benchmarkers.in_expansion.benchmark import Benchmarker

from string_constants import AB3P_DATASET, SH_DATASET, BIOADI_DATASET, MEDSTRACT_DATASET

test_dataset_name = BIOADI_DATASET

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = create_configargparser(
        in_expander=True,
        external_data=True,
        results_db_config=True,
    )
    args = parser.parse_args()

    if args.external_data:
        in_expander_train_dataset_name = [AB3P_DATASET, SH_DATASET, BIOADI_DATASET, MEDSTRACT_DATASET]
    else:
        in_expander_train_dataset_name = BIOADI_DATASET

    benchmarker = Benchmarker(
        in_expander_name=args.in_expander,
        in_expander_args=args.in_expander_args,
        in_expander_train_dataset_names=in_expander_train_dataset_name,
        in_expander_test_dataset_name=test_dataset_name,
        results_report=ResultsReporter,
    )

    generatedFilesFolder = getDatasetGeneratedFilesPath(test_dataset_name)

    # currently the test_articles list holds all of the articles ids from the dataset
    # if you choose to train with the same dataset as the test dataset
    # please remove a percentage of these articles from test
    test_articles = pickle.load(
        open(generatedFilesFolder + "test_articles.pickle", "rb")
    )

    # no actual need to specify the train articles
    # if we choose the same dataset for train and test
    # the system will just remove the articles that are in test_articles
    # from the train dataset
    train_articles = pickle.load(
        open(generatedFilesFolder + "train_articles.pickle", "rb")
    )

    indexedFolds = [(test_articles, train_articles)]

    benchmarker.run(
        indexedFolds,
        report_db_config=args.results_database_configuration,
        num_processes=len(indexedFolds),
    )
