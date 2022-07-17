"""
Input classes for acronym expander

Includes Input articles with acronyms to expand and train data
"""
from enum import Enum
import os
from typing import Optional, Callable, Union

from sqlitedict import SqliteDict

from DataCreators import AcronymDB, ArticleDB
from DataCreators.ArticleAcronymDB import getArticleAcronymDB
from Logger import logging
from helper import (
    getDatasetGeneratedFilesPath,
    get_acronym_db_path,
    get_raw_article_db_path,
    get_preprocessed_article_db_path,
)
from string_constants import USERS_WIKIPEDIA


logger = logging.getLogger(__name__)

StorageType = Enum("StorageType", ("memory", "SQLITE"))


class InputArticle:
    """
    Input article that contains text (either raw or preprocessed)
    """

    # TODO add types to dict when upgrade to python 3.9
    def __init__(  # pylint: disable=too-many-arguments
        self,
        raw_text: Optional[str] = None,
        preprocesser: Optional[Callable] = None,
        article_id: Optional[Union[str, int]] = None,
        raw_articles_db: Optional[dict] = None,
        preprocessed_articles_db: Optional[dict] = None,
    ):
        """
        Creates an Input Article by one of these options:
            - Either the raw text or the article_id and raw_articles_db is provided
            - Either the proprocessor or the article_id and preprocessed_articles_db is provided

        Options using databases are used manly during benchmarking, where the Input articles are
         stored in Databases

        Args:
            raw_text: article plain text
            preprocesser: text preprocessing function
            article_id: article identifier
            raw_articles_db: articles database containing article ids and raw plain text
            preprocessed_articles_db: articles database containing article ids and preprocessed
                plain text
        """
        self.raw_text = raw_text
        self.preprocesser = preprocesser
        self.article_id = article_id
        self.raw_articles_db = raw_articles_db
        self.preprocessed_articles_db = preprocessed_articles_db

        self.preprocessed_text = None

    def set_raw_text(self, raw_text: str):
        """
        Args:
            raw_text: article raw text
        """
        self.raw_text = raw_text
        self.preprocessed_text = None

    def get_raw_text(self) -> str:
        """
        Returns:
            article raw text
        """
        if not self.raw_text:
            self.raw_text = self.raw_articles_db[self.article_id]
        return self.raw_text

    def set_preprocessor(self, preprocessor: Callable):
        """
        Args:
            preprocessor: text preprocessing function
        """
        self.preprocesser = preprocessor

    def get_preprocessed_text(self) -> str:
        """
        Returns:
             article preprocessed text
        """
        if not self.preprocessed_text:
            if self.preprocessed_articles_db:
                self.preprocessed_text = self.preprocessed_articles_db[self.article_id]
            else:
                self.preprocessed_text = self.preprocesser(self.raw_text)
        return self.preprocessed_text


class TrainInDataManager:
    """
    Manager for the training data
    """

    def __init__(
        self,
        dataset_name: str,
        storage_type: StorageType = "SQLITE",
    ):
        """
        Args:
            dataset_name: name of the dataset to manage
            storage_type: database storage type
        """

        self.dataset_name = dataset_name
        self.storage_type = storage_type

        if storage_type == "memory":
            self.articles_raw_db = {}
            self.article_acronym_db = {}
        else:
            article_raw_db_path = get_raw_article_db_path(dataset_name)
            self.articles_raw_db = ArticleDB.load(
                path=article_raw_db_path, storageType=self.storage_type
            )
            self.article_acronym_db = getArticleAcronymDB(dataset_name)

            for article_id, acro_exp_dict in self.article_acronym_db.items():
                tmp_acro_exp_dict = {}
                only_in_expansions = False
                if len(acro_exp_dict) == 0:
                    continue
                for acro, exp in acro_exp_dict.items():
                    # checking if the dataset includes a flag
                    # for expansions in and not in text
                    if exp != None and len(exp) == 2:
                        if exp[1]:
                            tmp_acro_exp_dict[acro] = exp[0]
                        else:
                            tmp_acro_exp_dict[acro] = None
                    else:
                        only_in_expansions = True
                        break
                if only_in_expansions:
                    break
                self.article_acronym_db[article_id] = tmp_acro_exp_dict

    def get_raw_articles_db(self) -> dict:
        """
        Returns:
            dict like database that contains articles raw text as article_id:article_raw_text
        """
        return self.articles_raw_db

    def get_article_acronym_db(self) -> dict:
        """
        Returns:
            dict like database that contains acronym expansion pairs by article as
            article_id:{acronym:expansion}
        """
        return self.article_acronym_db

    def get_dataset_name(self) -> str:
        """
        Returns:
            str: the training dataset name
        """
        return self.dataset_name

    def close(self):
        """
        Closes the managed databases
        """
        all_dbs = [
            self.articles_raw_db,
            self.article_acronym_db,
        ]

        for db in all_dbs:
            if hasattr(db, "close"):
                db.close()

    def delete(self):
        """
        Deletes all managed databases
        """

        self.close()

        all_dbs = [
            self.articles_raw_db,
            self.article_acronym_db,
        ]

        for db in all_dbs:
            if isinstance(db, SqliteDict):
                if os.path.isfile(db.filename):
                    os.remove(db.filename)
            else:
                del db


class TrainOutDataManager(TrainInDataManager):
    """
    Manager for the training data
    """

    def __init__(
        self,
        dataset_name: str,
        fold: Optional[Union[str, int]] = None,
        storage_type: StorageType = "SQLITE",
    ):
        """
        Args:
            dataset_name: name of the dataset to manage
            fold: fold number
            storage_type: database storage type
        """

        self.dataset_name = dataset_name
        self.fold = fold
        self.storage_type = storage_type

        if storage_type == "memory":
            self.acronym_db = {}
            self.articles_preprocessed_db = {}
            self.articles_raw_db = {}
            self.article_acronym_db = {}
        else:
            if fold is not None:
                dbs_file_name = (
                    getDatasetGeneratedFilesPath(dataset_name)
                    + "db_"
                    + str(fold)
                    + "_%s.sqlite"
                )
                self.acronym_db = SqliteDict(dbs_file_name % "TrainAcronyms", flag="r")
                self.article_acronym_db = SqliteDict(
                    dbs_file_name % "TrainArticleAcronyms", flag="r"
                )
                self.articles_raw_db = SqliteDict(
                    dbs_file_name % "TrainRawArticles", flag="r"
                )
                a = self.articles_raw_db.conn.execute("PRAGMA cache_size=-10;")

                self.articles_preprocessed_db = SqliteDict(
                    dbs_file_name % "TrainPreprocessedArticles", flag="r"
                )
                a = self.articles_preprocessed_db.conn.execute("PRAGMA cache_size=-10;")
                logger.critical("Cache size: %s", str(a))

            else:

                article_raw_db_path = get_raw_article_db_path(dataset_name)
                self.articles_raw_db = ArticleDB.load(
                    path=article_raw_db_path, storageType="SQLite"
                )
                article_preprocssed_db_path = get_preprocessed_article_db_path(
                    dataset_name
                )
                self.articles_preprocessed_db = ArticleDB.load(
                    path=article_preprocssed_db_path, storageType="SQLite"
                )
                acronym_db_path = get_acronym_db_path(dataset_name)
                self.acronym_db = AcronymDB.load(
                    path=acronym_db_path, storageType="SQLite"
                )
                self.article_acronym_db = getArticleAcronymDB(dataset_name)

    def get_fold(self) -> Union[str, int]:
        """
        Returns:
            Fold number or name identifier
        """
        return self.fold

    def get_acronym_db(self) -> dict:
        """
        Returns:
            dict like database that contains acronym and list of available expansions with
            article id where it appears as acronym:[(expansion1,article),...]
        """
        return self.acronym_db

    def get_raw_articles_db(self) -> dict:
        """
        Returns:
            dict like database that contains articles raw text as article_id:article_raw_text
        """
        return self.articles_raw_db

    def get_preprocessed_articles_db(self) -> dict:
        """
        Returns:
            dict like database that contains articles preprocessed text as
            article_id:article_preprocessed_text
        """
        return self.articles_preprocessed_db

    def get_article_acronym_db(self) -> dict:
        """
        Returns:
            dict like database that contains acronym expansion pairs by article as
            article_id:{acronym:expansion}
        """
        return self.article_acronym_db

    def close(self):
        """
        Closes the managed databases
        """
        all_dbs = [
            self.acronym_db,
            self.articles_raw_db,
            self.articles_preprocessed_db,
            self.article_acronym_db,
        ]

        for db in all_dbs:
            if hasattr(db, "close"):
                db.close()

    def delete(self):
        """
        Deletes all managed databases
        """

        self.close()

        all_dbs = [
            self.acronym_db,
            self.articles_raw_db,
            self.articles_preprocessed_db,
            self.article_acronym_db,
        ]

        for db in all_dbs:
            if isinstance(db, SqliteDict):
                if os.path.isfile(db.filename):
                    os.remove(db.filename)
            else:
                del db
