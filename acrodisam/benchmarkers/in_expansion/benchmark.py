"""Bennchmark for in-expansion.
@author : jpereira
"""

import os
from sqlitedict import SqliteDict
from DataCreators import ArticleDB
from DataCreators.ArticleAcronymDB import getArticleAcronymDB
from acronym_expander import AcronymExpanderFactory
from benchmarkers.benchmark_base import BenchmarkerBase
from Logger import logging
from helper import ExecutionTimeObserver, get_raw_article_db_path
from inputters import TrainInDataManager, TrainOutDataManager
from run_config import RunConfig


logger = logging.getLogger(__name__)


class Benchmarker(BenchmarkerBase):
    def __init__(
        self,
        in_expander_name,
        in_expander_args,
        in_expander_train_dataset_names,
        in_expander_test_dataset_name,
        results_report,
    ):
        experiment_name = (
            "TestIn="
            + in_expander_test_dataset_name
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

        experiment_parameters = {
            "in_expander": in_expander_name,
            "in_expander_args": in_expander_args,
        }

        self.in_expander_train_dataset_names = in_expander_train_dataset_names

        self.expander_factory = AcronymExpanderFactory(
            in_expander_name=in_expander_name,
            in_expander_args=in_expander_args,
            run_config=run_config,
        )

        super().__init__(
            run_config=run_config,
            experiment_parameters=experiment_parameters,
            train_dataset_name="",
            test_dataset_name=in_expander_test_dataset_name,
            results_report=results_report,
        )

    def _create_expander(self, train_data_manager: TrainInDataManager):
        return self.expander_factory._get_in_expander(train_data_manager)

    def _get_in_expander_test_data(
        self, test_dataset_name: str, test_articles_ids: list
    ):
        test_article_acronym_db = getArticleAcronymDB(self.test_dataset_name)
        test_article_raw_db_path = get_raw_article_db_path(self.test_dataset_name)
        test_articles_raw_db = ArticleDB.load(
            path=test_article_raw_db_path, storageType="pickle"
        )

        for article_id, acro_exp_dict in list(test_article_acronym_db.items()):
            if article_id not in test_articles_ids:
                test_article_acronym_db.pop(article_id)
                test_articles_raw_db.pop(article_id)
            else:
                if len(acro_exp_dict) != 0:
                    tmp_acro_exp_dict = self._remove_in_expansion_out_expansion_flag(
                        acro_exp_dict
                    )

                    if len(tmp_acro_exp_dict) == 0:
                        continue
                    test_article_acronym_db[article_id] = tmp_acro_exp_dict

        return test_article_acronym_db, test_articles_raw_db

    def _evaluate(
        self,
        results_reporter,
        test_articles_ids,
        train_articles_ids=None,
        fold="TrainData",
    ):

        deleteOnClose = False

        in_expander_train_data_manager = TrainInDataManager(
            self.run_config.name, storage_type="memory"
        )

        self._load_dbs_for_in_expansion(
            in_expander_train_dataset_names=self.in_expander_train_dataset_names,
            test_dataset_name=self.test_dataset_name,
            test_articles_ids=test_articles_ids,
            in_expander_train_data_manager=in_expander_train_data_manager,
        )

        test_article_acronym_db, test_articles_raw_db = self._get_in_expander_test_data(
            self.test_dataset_name, test_articles_ids
        )

        try:
            logger.info("Creating the In-Expander")

            model_train_time = ExecutionTimeObserver()
            model_train_time.start()

            in_expander = self._create_expander(in_expander_train_data_manager)

            model_train_time.stop()

            results_reporter.addModelExecutionTime(fold, model_train_time)

            logger.critical("Evaluating In-Expander test performance")

            for article_id, actual_expansions in test_article_acronym_db.items():
                logger.debug("articleID: %s", str(article_id))

                try:
                    article = test_articles_raw_db.get(article_id)
                    testInstanceExecutionTime = ExecutionTimeObserver()
                    testInstanceExecutionTime.start()
                    predicted_expansions = in_expander.get_acronym_expansion_pairs(
                        article
                    )
                    testInstanceExecutionTime.stop()

                    results_reporter.addTestResult(
                        fold,
                        article_id,
                        actual_expansions,
                        predicted_expansions,
                        testInstanceExecutionTime,
                    )

                except Exception as inst:  # pylint: disable=broad-except
                    logger.exception(
                        "skipping articleID: %s, error details:", str(article_id)
                    )
                    results_reporter.addTestError(fold, article_id, inst)

        finally:
            all_dbs = [
                in_expander_train_data_manager,
                test_article_acronym_db,
                test_articles_raw_db,
            ]
            for db in all_dbs:
                if hasattr(db, "close"):
                    db.close()
                if deleteOnClose:
                    if isinstance(db, SqliteDict):
                        if os.path.isfile(db.filename):
                            os.remove(db.filename)
                    elif isinstance(db, TrainOutDataManager) or isinstance(
                        db, TrainInDataManager
                    ):
                        db.delete()
                    else:
                        del db

        return fold, True

    def _process_article(self, expander, article_for_testing, article_id, acronyms):
        pass
