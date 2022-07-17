import abc
from concurrent.futures import ProcessPoolExecutor
import functools
import os

import multiprocessing_logging
from sqlitedict import SqliteDict

from DataCreators import AcronymDB, ArticleDB, ArticleAcronymDB
from DataCreators.ArticleAcronymDB import (
    create_article_acronym_db_from_acronym_db,
    getArticleAcronymDB,
)
from DatasetParsers.expansion_linkage import (
    _resolve_exp_acronym_db,
    replace_exp_in_article,
)
from Logger import logging
from acronym_expander import InputArticle
from helper import (
    get_acronym_db_path,
    getArticleAcronymDBPath,
    getDatasetGeneratedFilesPath,
    ExecutionTimeObserver,
    get_raw_article_db_path,
    get_preprocessed_article_db_path,
)
from inputters import TrainInDataManager, TrainOutDataManager


logger = logging.getLogger(__name__)

MULTIPLE_DATASET_PREFIX = "%s_"


class BenchmarkerBase(metaclass=abc.ABCMeta):
    def __init__(
        self,
        run_config,
        experiment_parameters,
        train_dataset_name,
        test_dataset_name,
        results_report,
        expansion_linkage=False,
    ):
        self.run_config = run_config
        self.experiment_parameters = experiment_parameters
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.results_report = results_report
        self.expansion_linkage = expansion_linkage

    def _proxy_function(self, *args, **kwargs):
        logger_func = logging.getLogger(__name__ + "._proxy_function")
        if len(args) > 3:
            fold = args[3]
        else:
            fold = "TrainData"
        try:
            fold, success = self._evaluate(*args, **kwargs)

            status = "successfully." if success else " failed."
            logger_func.critical(
                "Subprocess executing fold %s executed %s", str(fold), status
            )
        except BaseException as base_exp:
            logger_func.critical(
                "Fatal error in subprocess executing fold %s", fold, exc_info=True
            )
            logger_func.critical(base_exp.args)
            raise base_exp

    def run(
        self,
        indexed_folds,
        save_results_per_article=None,
        report_db_config=None,
        num_processes=1,
    ):

        """We save predictions per document if not doing
        cross-validation experiments to setup hyperparameters
         This is usefull for statistical testing
        """
        if save_results_per_article is None:
            save_results_per_article = len(indexed_folds) < 2

        multiprocessing_logging.install_mp_handler(logger)

        with self.results_report(
            experiment_name=self.run_config.name,
            experiment_parameters=self.experiment_parameters,
            save_results_per_article=save_results_per_article,
            db_config=report_db_config,
        ) as results_reporter:
            if num_processes > 1 and len(indexed_folds) > 1:

                partial_proxy_func = functools.partial(
                    self._proxy_function, results_reporter
                )
                with ProcessPoolExecutor(max_workers=num_processes) as executer:
                    executer.map(partial_proxy_func, *list(zip(*indexed_folds)))
            else:
                for index_fold in indexed_folds:
                    self._proxy_function(results_reporter, *index_fold)

    def _get_realiged_acronym_db_aux(
        self, article_ids_to_keep, dataset_name, new_acronym_db, id_prefix=None
    ):
        acronym_db = AcronymDB.load(path=get_acronym_db_path(dataset_name))

        for acronym in acronym_db.keys():
            validEntries = []
            for entry in acronym_db[acronym]:
                if entry[1] in article_ids_to_keep:
                    if id_prefix is None:
                        validEntries.append(entry)
                    else:
                        validEntries.append((entry[0], id_prefix + entry[1]))
            current_exp = new_acronym_db.get(acronym, [])
            current_exp.extend(validEntries)
            new_acronym_db[acronym] = current_exp

    def _get_realigned_acronym_db(
        self, article_ids_to_keep, dataset_name, new_acronym_db=None
    ):
        if new_acronym_db is None:
            new_acronym_db = {}

        if isinstance(dataset_name, list):
            for dataset_id, (single_article_ids_to_keep, name) in enumerate(
                zip(article_ids_to_keep, dataset_name)
            ):
                self._get_realiged_acronym_db_aux(
                    single_article_ids_to_keep,
                    name,
                    new_acronym_db,
                    id_prefix=MULTIPLE_DATASET_PREFIX % dataset_id,
                )

        else:
            self._get_realiged_acronym_db_aux(
                article_ids_to_keep, dataset_name, new_acronym_db
            )
        return new_acronym_db

    def _get_realigned_article_acronym_db_aux(
        self, articles_ids_to_keep, dataset_name, new_article_acronym_db, id_prefix=None
    ):
        article_acronym_db = ArticleAcronymDB.load(
            getArticleAcronymDBPath(dataset_name)
        )

        for article_id in articles_ids_to_keep:
            acronymExpansions = article_acronym_db.get(article_id)

            if acronymExpansions:
                if id_prefix is None:
                    new_article_acronym_db[article_id] = acronymExpansions
                else:
                    new_article_acronym_db[id_prefix + article_id] = acronymExpansions
            else:
                logger.warning(
                    "No expansions found for article: %s in dataset: %s",
                    str(article_id),
                    dataset_name,
                )

    def _get_realigned_article_acronym_db(
        self, articles_ids_to_keep, dataset_name, new_article_acronym_db
    ):
        if isinstance(dataset_name, list):
            for dataset_id, (single_article_ids_to_keep, name) in enumerate(
                zip(articles_ids_to_keep, dataset_name)
            ):
                self._get_realigned_article_acronym_db_aux(
                    single_article_ids_to_keep,
                    name,
                    new_article_acronym_db,
                    id_prefix=MULTIPLE_DATASET_PREFIX % dataset_id,
                )
        else:
            self._get_realigned_article_acronym_db_aux(
                articles_ids_to_keep, dataset_name, new_article_acronym_db
            )

        return new_article_acronym_db

    def _verifyTrainSet(self, trainArticlesIDs, acronymDB, testArticleIDs):
        for articleId in trainArticlesIDs:
            if articleId in testArticleIDs:
                return False
        for acronym in acronymDB:
            for _, articleId in acronymDB[acronym]:
                if articleId in testArticleIDs:
                    return False
        return True

    def _load_articles_train_test_sets(
        self,
        test_article_ids,
        train_article_ids,
        dataset_name,
        train_articles,
        test_articles,
        test_article_acronyms,
        preprocessed: bool,
        exp_changes_articles=None,
        results_reporter=None,
        fold=None,
    ):

        if preprocessed:
            article_db_path = get_preprocessed_article_db_path(dataset_name)
        else:
            article_db_path = get_raw_article_db_path(dataset_name)

        article_db = ArticleDB.load(path=article_db_path)

        for article_id, text in article_db.items():
            text = replace_exp_in_article(article_id, text, exp_changes_articles)
            if article_id in test_article_ids:
                try:
                    acronym_expansions = test_article_acronyms[article_id]
                    text, existing_acronym_expansions = self._preprocess_test_article(
                        text, acronym_expansions
                    )
                    test_article_acronyms[article_id] = existing_acronym_expansions
                    test_articles[article_id] = text
                except Exception as inst:  # pylint: disable=broad-except
                    logger.exception(
                        "skipping articleID: %s, error details:", str(article_id)
                    )
                    results_reporter.addTestError(fold, article_id, inst)
            elif article_id in train_article_ids:
                train_articles[article_id] = text
            else:
                pass

    def _load_articles_single_dataset(
        self,
        articles_ids_to_keep,
        dataset_name,
        new_articles,
        preprocessed: bool,
        exp_changes_articles=None,
        id_prefix=None,
        test_article_acronyms=None,  # None means its training data
        results_reporter=None,
        fold=None,
    ):

        if preprocessed:
            article_db_path = get_preprocessed_article_db_path(dataset_name)
        else:
            article_db_path = get_raw_article_db_path(dataset_name)

        article_db = ArticleDB.load(path=article_db_path)

        for article_id, text in article_db.items():
            if article_id in articles_ids_to_keep:
                if id_prefix is None:
                    text = replace_exp_in_article(
                        article_id, text, exp_changes_articles
                    )
                else:
                    text = replace_exp_in_article(
                        id_prefix + article_id, text, exp_changes_articles
                    )

                if test_article_acronyms:
                    try:
                        acronym_expansions = test_article_acronyms.get(article_id, {})
                        (
                            text,
                            existing_acronym_expansions,
                        ) = self._preprocess_test_article(text, acronym_expansions)
                        test_article_acronyms[article_id] = existing_acronym_expansions
                    except Exception as inst:  # pylint: disable=broad-except
                        logger.exception(
                            "skipping articleID: %s, error details:", str(article_id)
                        )
                        results_reporter.addTestError(fold, article_id, inst)

                if id_prefix is None:
                    new_articles[article_id] = text
                else:
                    new_articles[id_prefix + article_id] = text

    def _load_articles(
        self,
        train_article_ids,
        test_article_ids,
        train_raw_articles,
        train_preprocessed_articles,
        test_raw_articles,
        test_preprocessed_articles,
        test_article_acronyms,
        exp_changes_articles,
        results_reporter,
        fold,
    ):

        logger.info("Staring loading articles")

        if self.train_dataset_name == self.test_dataset_name:
            self._load_articles_train_test_sets(
                test_article_ids=test_article_ids,
                train_article_ids=train_article_ids,
                dataset_name=self.train_dataset_name,
                train_articles=train_raw_articles,
                test_articles=test_raw_articles,
                test_article_acronyms=test_article_acronyms,
                preprocessed=False,
                exp_changes_articles=exp_changes_articles,
                results_reporter=results_reporter,
                fold=fold,
            )

            self._load_articles_train_test_sets(
                test_article_ids=test_article_ids,
                train_article_ids=train_article_ids,
                dataset_name=self.train_dataset_name,
                train_articles=train_preprocessed_articles,
                test_articles=test_preprocessed_articles,
                test_article_acronyms=test_article_acronyms,
                preprocessed=True,
                exp_changes_articles=exp_changes_articles,
                results_reporter=results_reporter,
                fold=fold,
            )

        else:
            if isinstance(self.train_dataset_name, list):
                for dataset_id, (single_article_ids_to_keep, name) in enumerate(
                    zip(train_article_ids, self.train_dataset_name)
                ):
                    self._load_articles_single_dataset(
                        single_article_ids_to_keep,
                        name,
                        train_raw_articles,
                        preprocessed=False,
                        exp_changes_articles=exp_changes_articles,
                        id_prefix=MULTIPLE_DATASET_PREFIX % dataset_id,
                    )

                    self._load_articles_single_dataset(
                        single_article_ids_to_keep,
                        name,
                        train_preprocessed_articles,
                        preprocessed=True,
                        exp_changes_articles=exp_changes_articles,
                        id_prefix=MULTIPLE_DATASET_PREFIX % dataset_id,
                    )
            else:
                self._load_articles_single_dataset(
                    train_article_ids,
                    self.train_dataset_name,
                    train_raw_articles,
                    preprocessed=False,
                    exp_changes_articles=exp_changes_articles,
                )

                self._load_articles_single_dataset(
                    train_article_ids,
                    self.train_dataset_name,
                    train_preprocessed_articles,
                    preprocessed=True,
                    exp_changes_articles=exp_changes_articles,
                )

            self._load_articles_single_dataset(
                test_article_ids,
                self.test_dataset_name,
                test_raw_articles,
                preprocessed=False,
                test_article_acronyms=test_article_acronyms,
                results_reporter=results_reporter,
                fold=fold,
            )

            try:
                self._load_articles_single_dataset(
                    test_article_ids,
                    self.test_dataset_name,
                    test_preprocessed_articles,
                    preprocessed=True,
                    test_article_acronyms=test_article_acronyms,
                    results_reporter=results_reporter,
                    fold=fold,
                )
            except Exception:
                logger.info("Preprocessed test articles not loaded")

        logger.info("Finished putting articles into DBs")
        return (
            train_raw_articles,
            train_preprocessed_articles,
            test_raw_articles,
            test_preprocessed_articles,
        )

    def _load_acronyms_dbs(
        self,
        train_articles_ids,
        test_articles_ids,
        train_acronyms,
        train_article_acronyms,
        test_article_acronyms,
    ):
        logger.info("correcting acronymDB")
        if self.expansion_linkage:
            train_acronym_db_no_linkage = self._get_realigned_acronym_db(
                train_articles_ids, self.train_dataset_name
            )
            exp_changes_articles = _resolve_exp_acronym_db(
                train_acronym_db_no_linkage, train_acronyms
            )
            create_article_acronym_db_from_acronym_db(
                train_acronyms, train_article_acronyms
            )
        else:
            self._get_realigned_acronym_db(
                train_articles_ids, self.train_dataset_name, train_acronyms
            )
            exp_changes_articles = None
            self._get_realigned_article_acronym_db(
                train_articles_ids, self.train_dataset_name, train_article_acronyms
            )

        self._get_realigned_article_acronym_db(
            test_articles_ids, self.test_dataset_name, test_article_acronyms
        )

        logger.info("verifying training dataset")
        if not self._verifyTrainSet(
            train_articles_ids, train_acronyms, test_articles_ids
        ):
            logger.error(
                "verification of train datasets failed for articleIDs %s",
                str(test_articles_ids),
            )
            # return fold_num, False

        return exp_changes_articles

    def _load_dbs(
        self,
        train_articles_ids,
        test_articles_ids,
        train_acronyms,
        train_article_acronyms,
        test_article_acronyms,
        train_raw_articles,
        train_preprocessed_articles,
        test_raw_articles,
        test_preprocessed_articles,
        results_reporter,
        fold,
    ):

        exp_changes_articles = self._load_acronyms_dbs(
            train_articles_ids,
            test_articles_ids,
            train_acronyms,
            train_article_acronyms,
            test_article_acronyms,
        )

        if exp_changes_articles is None:
            exp_changes_articles = {}

        self._load_articles(
            train_articles_ids,
            test_articles_ids,
            train_raw_articles,
            train_preprocessed_articles,
            test_raw_articles,
            test_preprocessed_articles,
            test_article_acronyms,
            exp_changes_articles,
            results_reporter,
            fold,
        )

        return (
            train_acronyms,
            train_article_acronyms,
            train_raw_articles,
            train_preprocessed_articles,
            test_article_acronyms,
            test_raw_articles,
            test_preprocessed_articles,
        )

    def _load_dbs_to_sqlite(
        self, train_articles_ids, test_articles_ids, results_reporter, fold=0
    ):

        dbs_file_name = (
            getDatasetGeneratedFilesPath(self.run_config.name)
            + "db_"
            + str(fold)
            + "_%s.sqlite"
        )

        # build DBs if not exist
        if not os.path.exists(
            dbs_file_name % "TestPreprocessedArticles"
        ):  # we test the last to be created
            with SqliteDict(
                dbs_file_name % "TrainAcronyms", flag="n", autocommit=True
            ) as train_acronyms, SqliteDict(
                dbs_file_name % "TrainArticleAcronyms", flag="n", autocommit=True
            ) as train_article_acronyms, SqliteDict(
                dbs_file_name % "TestArticleAcronyms", flag="n", autocommit=True
            ) as test_article_acronyms, SqliteDict(
                dbs_file_name % "TrainRawArticles", flag="n", autocommit=True
            ) as train_raw_articles, SqliteDict(
                dbs_file_name % "TrainPreprocessedArticles", flag="n", autocommit=True
            ) as train_preprocessed_articles, SqliteDict(
                dbs_file_name % "TestRawArticles", flag="n", autocommit=True
            ) as test_raw_articles, SqliteDict(
                dbs_file_name % "TestPreprocessedArticles", flag="n", autocommit=True
            ) as test_preprocessed_articles:

                self._load_dbs(
                    train_articles_ids=train_articles_ids,
                    test_articles_ids=test_articles_ids,
                    train_acronyms=train_acronyms,
                    train_article_acronyms=train_article_acronyms,
                    test_article_acronyms=test_article_acronyms,
                    train_raw_articles=train_raw_articles,
                    train_preprocessed_articles=train_preprocessed_articles,
                    test_raw_articles=test_raw_articles,
                    test_preprocessed_articles=test_preprocessed_articles,
                    results_reporter=results_reporter,
                    fold=fold,
                )
                logger.info("Closing DBs")

        # load sqlite
        train_data_manager = TrainOutDataManager(self.run_config.name, fold)

        test_article_acronym_db = SqliteDict(
            dbs_file_name % "TestArticleAcronyms", flag="r"
        )
        test_raw_articles_db = SqliteDict(dbs_file_name % "TestRawArticles", flag="r")
        test_preprocessed_articles_db = SqliteDict(
            dbs_file_name % "TestPreprocessedArticles", flag="r"
        )

        return (
            train_data_manager,
            test_article_acronym_db,
            test_raw_articles_db,
            test_preprocessed_articles_db,
        )

    def _remove_in_expansion_out_expansion_flag(self, acro_exp_dict):
        """
        Removes the flag for each expansion in the acronym-expansion dictionary
        that indicates if an expansion is present in the article text or not.

        Args:
            acro_exp_dict (dict):
             a dictionary where each key is an acronym and each value is a tuple (expansion, flag) where the
             flag is a boolean that indicates if the expansion is present in the article text or not.

        Returns:
            dict: a dictionary equal to acro_exp_dict but with the flag that indicates if an expansion is present
            in text or not, removed.
        """
        tmp_acro_exp_dict = {}
        for acro, exp in acro_exp_dict.items():
            # checking if the dataset includes a flag
            # for expansions in and not in text
            if exp != None and len(exp) == 2:
                if exp[1]:
                    tmp_acro_exp_dict[acro] = exp[0]
                else:
                    tmp_acro_exp_dict[acro] = None
            else:
                return tmp_acro_exp_dict

        return tmp_acro_exp_dict

    def _load_dbs_for_in_expansion(
        self,
        in_expander_train_dataset_names,
        test_dataset_name,
        test_articles_ids,
        in_expander_train_data_manager,
    ):
        if isinstance(in_expander_train_dataset_names, list):
            for dataset_name in in_expander_train_dataset_names:

                article_raw_db_path = get_raw_article_db_path(dataset_name)
                tmp_articles_raw_db = ArticleDB.load(
                    path=article_raw_db_path, storageType="pickle"
                )
                tmp_article_acronym_db = getArticleAcronymDB(dataset_name)

                if dataset_name == test_dataset_name:
                    for article_id, acro_exp_dict in list(
                        tmp_article_acronym_db.items()
                    ):
                        if article_id in test_articles_ids:
                            tmp_article_acronym_db.pop(article_id)
                            tmp_articles_raw_db.pop(article_id)

                for article_id, acro_exp_dict in tmp_article_acronym_db.items():
                    if len(acro_exp_dict) != 0:
                        tmp_acro_exp_dict = (
                            self._remove_in_expansion_out_expansion_flag(acro_exp_dict)
                        )
                        if len(tmp_acro_exp_dict) == 0:
                            break
                        tmp_article_acronym_db[article_id] = tmp_acro_exp_dict

                in_expander_train_data_manager.articles_raw_db = {
                    **in_expander_train_data_manager.articles_raw_db,
                    **tmp_articles_raw_db,
                }
                in_expander_train_data_manager.article_acronym_db = {
                    **in_expander_train_data_manager.article_acronym_db,
                    **tmp_article_acronym_db,
                }

        else:
            if in_expander_train_dataset_names == test_dataset_name:
                article_raw_db_path = get_raw_article_db_path(
                    in_expander_train_dataset_names
                )
                in_expander_train_data_manager.articles_raw_db = ArticleDB.load(
                    path=article_raw_db_path, storageType="pickle"
                )
                in_expander_train_data_manager.article_acronym_db = getArticleAcronymDB(
                    in_expander_train_dataset_names
                )

                for (
                    article_id,
                    acro_exp_dict,
                ) in list(in_expander_train_data_manager.article_acronym_db.items()):
                    if article_id in test_articles_ids:
                        in_expander_train_data_manager.article_acronym_db.pop(
                            article_id
                        )
                        in_expander_train_data_manager.articles_raw_db.pop(article_id)
                    else:
                        if len(acro_exp_dict) != 0:
                            tmp_acro_exp_dict = (
                                self._remove_in_expansion_out_expansion_flag(
                                    acro_exp_dict
                                )
                            )

                            if len(tmp_acro_exp_dict) == 0:
                                continue
                            in_expander_train_data_manager.article_acronym_db[
                                article_id
                            ] = tmp_acro_exp_dict

            else:
                article_raw_db_path = get_raw_article_db_path(
                    in_expander_train_dataset_names
                )
                in_expander_train_data_manager.articles_raw_db = ArticleDB.load(
                    path=article_raw_db_path, storageType="pickle"
                )
                in_expander_train_data_manager.article_acronym_db = getArticleAcronymDB(
                    in_expander_train_dataset_names
                )

                for (
                    article_id,
                    acro_exp_dict,
                ) in in_expander_train_data_manager.article_acronym_db.items():
                    if len(acro_exp_dict) != 0:
                        tmp_acro_exp_dict = (
                            self._remove_in_expansion_out_expansion_flag(acro_exp_dict)
                        )
                        if len(tmp_acro_exp_dict) == 0:
                            break
                        in_expander_train_data_manager.article_acronym_db[
                            article_id
                        ] = tmp_acro_exp_dict

    def _get_dbs(self, test_articles_ids, train_articles_ids, fold, results_reporter):

        if self.run_config.persistent_articles is None:
            train_data_manager = TrainOutDataManager(
                self.run_config.name, fold, storage_type="memory"
            )
            (
                train_acronyms,
                train_article_acronyms,
                train_raw_articles,
                train_preprocessed_articles,
                test_article_acronyms,
                test_raw_articles,
                test_preprocessed_articles,
            ) = self._load_dbs(
                train_articles_ids=train_articles_ids,
                test_articles_ids=test_articles_ids,
                train_acronyms=train_data_manager.get_acronym_db(),
                train_article_acronyms=train_data_manager.get_article_acronym_db(),
                test_article_acronyms={},
                train_raw_articles=train_data_manager.get_raw_articles_db(),
                train_preprocessed_articles=train_data_manager.get_preprocessed_articles_db(),
                test_raw_articles={},
                test_preprocessed_articles={},
                fold=fold,
                results_reporter=results_reporter,
            )

            return (
                train_data_manager,
                test_article_acronyms,
                test_raw_articles,
                test_preprocessed_articles,
            )

        if self.run_config.persistent_articles == "SQLITE":
            return self._load_dbs_to_sqlite(
                train_articles_ids=train_articles_ids,
                test_articles_ids=test_articles_ids,
                fold=fold,
                results_reporter=results_reporter,
            )

        raise ValueError(
            "persitentArticles value unkown: {}".format(
                self.run_config.persistent_articles
            )
        )

    @abc.abstractmethod
    def _create_expander(self, train_data_manager: TrainOutDataManager):
        pass

    @abc.abstractmethod
    def _process_article(self, expander, article_for_testing, article_id, acronyms):
        pass

    def _preprocess_test_article(self, article, actual_expansions):
        return article, actual_expansions

    # Returns foldNum and a boolean value that is true if this function executed successfully, false otherwise
    def _evaluate(
        self,
        results_reporter,
        test_articles_ids,
        train_articles_ids=None,
        fold="TrainData",
    ):
        """
        Takes in test articles and gives back a report on the prediction performance.

        Args:
        testArticles (dict): {articleID: article text}

        Returns:
        scores (dict): None or {"correct_expansions": <0.0 to 1.0>, "incorrect_expansions": <0.0 to 1.0>}
        report (list): None or [articleID, [[acronym (sorted order), correct expansion?, true expansion, predicted expansion, confidence]] ]
        """
        deleteOnClose = False
        (
            out_expander_train_data_manager,
            test_article_acronym_db,
            test_raw_articles,
            test_preprocessed_articles,
        ) = self._get_dbs(
            test_articles_ids=test_articles_ids,
            train_articles_ids=train_articles_ids,
            fold=fold,
            results_reporter=results_reporter,
        )

        in_expander_train_data_manager = None

        if hasattr(self, "in_expander_train_dataset_names"):
            in_expander_train_data_manager = TrainInDataManager(
                self.run_config.name, storage_type="memory"
            )

            self._load_dbs_for_in_expansion(
                in_expander_train_dataset_names=self.in_expander_train_dataset_names,
                test_dataset_name=self.test_dataset_name,
                test_articles_ids=test_articles_ids,
                in_expander_train_data_manager=in_expander_train_data_manager,
            )

        try:
            logger.info("Creating the Expander")

            if in_expander_train_data_manager != None:
                expander, modelExecutionTime = self._create_expander(
                    out_expander_train_data_manager, in_expander_train_data_manager
                )
            else:
                expander, modelExecutionTime = self._create_expander(
                    out_expander_train_data_manager
                )

            results_reporter.addModelExecutionTime(fold, modelExecutionTime)

            logger.critical("evaluating test performance")
            total_num_articles = len(test_article_acronym_db)
            
            for n_article, (article_id, actual_expansions) in enumerate(test_article_acronym_db.items()):
                logger.debug("article %d out of %d, article_id: %s", n_article, total_num_articles ,str(article_id))
                #if article_id < 2677:
                #    continue
                #    break
                try:
                    article_for_testing = InputArticle(
                        article_id=article_id,
                        raw_articles_db=test_raw_articles,
                        preprocessed_articles_db=test_preprocessed_articles,
                    )
                    acronyms = actual_expansions.keys()
                    testInstanceExecutionTime = ExecutionTimeObserver()
                    testInstanceExecutionTime.start()
                    predicted_expansions = self._process_article(
                        expander, article_for_testing, article_id, acronyms
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
            if in_expander_train_data_manager != None:
                all_dbs = [
                    in_expander_train_data_manager,
                    out_expander_train_data_manager,
                    test_article_acronym_db,
                    test_raw_articles,
                    test_preprocessed_articles,
                ]
            else:
                all_dbs = [
                    out_expander_train_data_manager,
                    test_article_acronym_db,
                    test_raw_articles,
                    test_preprocessed_articles,
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
