import pickle

from acronym_expander import AcronymExpanderFactory, RunConfig, TrainOutDataManager
from benchmarkers.benchmark_base import BenchmarkerBase
from benchmarkers.end_to_end.results_reporter import ResultsReporter
from DataCreators import ArticleDB
from DatasetParsers import FullWikipedia
from helper import (
    create_configargparser,
    getArticleDBPath,
    getDatasetGeneratedFilesPath, get_raw_article_db_path,
)
from inputters import TrainInDataManager
from Logger import logging
from sqlitedict import SqliteDict
from string_constants import (
    DB_WITH_LINKS_SUFFIX,
    FULL_WIKIPEDIA_DATASET,
    USERS_WIKIPEDIA,
)

logger = logging.getLogger(__name__)
test_dataset_name = USERS_WIKIPEDIA


class BenchmarkerUserWikipedia(BenchmarkerBase):
    def __init__(
        self,
        in_expander_name,
        in_expander_args,
        out_expander_name,
        out_expander_args,
        out_expander_train_dataset_names,
        in_expander_train_dataset_names,
        follow_links,
    ):

        persistent_articles = "SQLITE"
        # persistent_articles = None
        expansion_linkage = True
        experiment_name = (
            "Test="
            + test_dataset_name
            + ":"
            + "TrainOut="
            + (
                "_".join(out_expander_train_dataset_names)
                if isinstance(out_expander_train_dataset_names, list)
                else out_expander_train_dataset_names
            )
            + ":"
            + "TrainIn="
            + (
                "_".join(in_expander_train_dataset_names)
                if isinstance(in_expander_train_dataset_names, list)
                else in_expander_train_dataset_names
            )
        )

        run_config = RunConfig()
        run_config.name = experiment_name
        run_config.save_and_load = True
        run_config.persistent_articles = persistent_articles

        experiment_parameters = {
            "in_expander": in_expander_name,
            "in_expander_args": in_expander_args,
            "out_expander": out_expander_name,
            "out_expander_args": out_expander_args,
            "follow_links": 0,
        }

        if follow_links:
            experiment_parameters["follow_links"] = 1

        super().__init__(
            run_config=run_config,
            experiment_parameters=experiment_parameters,
            train_dataset_name=out_expander_train_dataset_names,
            test_dataset_name=test_dataset_name,
            results_report=ResultsReporter,
            expansion_linkage=expansion_linkage,
        )

        self.in_expander_train_dataset_names = in_expander_train_dataset_names

        if out_expander_name == "maddog" and out_expander_args and len(out_expander_args) > 0 and "True" in out_expander_args[0]:
            bypass_db = True
        else:
            bypass_db = False

        self.expander_factory = AcronymExpanderFactory(
            text_preprocessor=FullWikipedia.text_preprocessing,
            in_expander_name=in_expander_name,
            in_expander_args=in_expander_args,
            out_expander_name=out_expander_name,
            out_expander_args=out_expander_args,
            follow_links=follow_links,
            follow_links_cache=True,
            bypass_db=bypass_db,
            run_config=run_config,
        )

        self.follow_links = follow_links

        self.base_url = "https://en.wikipedia.org/wiki/"

        if follow_links:
            self.test_articles_with_links = self._get_articles_with_links(
                test_dataset_name
            )

    def _get_articles_with_links(self, dataset_name):
        test_articles_with_links_path = (
            getArticleDBPath(dataset_name) + DB_WITH_LINKS_SUFFIX
        )
        test_articles_with_links = ArticleDB.load(path=test_articles_with_links_path)
        return test_articles_with_links

    def _create_expander(
        self,
        out_expander_train_data_manager: TrainOutDataManager,
        in_expander_train_data_manager: TrainInDataManager,
    ):
        acronym_expander, model_execution_time = self.expander_factory.create_expander(
            out_expander_train_data_manager, in_expander_train_data_manager
        )
        return acronym_expander, model_execution_time

    def _process_article(self, expander, article_for_testing, article_id, _):
        if self.follow_links:
            article_with_links = self.test_articles_with_links[article_id]
        else:
            article_with_links = None

        return expander.process_article(
            article_for_testing,
            test_article_id=article_id,
            text_with_links=article_with_links,
            base_url=self.base_url,
        )


if __name__ == "__main__":

    parser = create_configargparser(
        out_expander=True,
        in_expander=True,
        links=True,
        results_db_config=True,
    )
    args = parser.parse_args()

    if args.in_expander == "maddog":
        dataset_prefix = "_MadDog"
    else:
        dataset_prefix = ""

    out_expander_train_dataset_name = FULL_WIKIPEDIA_DATASET + dataset_prefix
    in_expander_train_dataset_name = "Ab3P-BioC"
    test_dataset_name = USERS_WIKIPEDIA

    benchmarker = BenchmarkerUserWikipedia(
        in_expander_name=args.in_expander,
        in_expander_args=args.in_expander_args,
        out_expander_name=args.out_expander,
        out_expander_args=args.out_expander_args,
        out_expander_train_dataset_names=out_expander_train_dataset_name,
        in_expander_train_dataset_names=in_expander_train_dataset_name,
        follow_links=args.follow_links,
    )

    generatedFilesFolderFullWikipedia = getDatasetGeneratedFilesPath(
        out_expander_train_dataset_name
    )


    testArticles = set(ArticleDB.load(get_raw_article_db_path(test_dataset_name)).keys())

    with SqliteDict(
        getArticleDBPath(FULL_WIKIPEDIA_DATASET), flag="r", autocommit=True
    ) as articleDB:
        wikiArticles = set(articleDB.keys())

    trainArticles = wikiArticles - testArticles

    indexed_folds = [(testArticles, trainArticles)]
    num_processes = 1

    benchmarker.run(
        indexed_folds,
        report_db_config=args.results_database_configuration,
        num_processes=num_processes,
    )
