"""



@author: jpereira
"""
from abc import ABCMeta, abstractmethod
import collections
import copy
import csv
from enum import Enum
import json
from multiprocessing import Manager
import platform
from threading import Thread
from inputters import TrainInDataManager

from mysql.connector import errorcode
import mysql.connector
from numpy import mean

from Logger import logging
from helper import ExecutionTimeObserver, flatten_to_str_list
from string_constants import FOLDER_LOGS

FILE_FAILED_DB_STATEMENTS = FOLDER_LOGS + "failed_db_statements.txt"

logger = logging.getLogger(__name__)

BASE_FIELDS = {
    "Processed folds": "int",
    "Total folds": "int",
    "Total articles": "int",
    "Skipped articles": "int",
}

EXECUTION_TIME_FIELDS = {
    "Model Execution Times": "BIGINT",
    "Average execution times per article": "DECIMAL(18,10)",
}
EXECUTION_TIME_PER_ARTICLES = {"Execution Times": "DECIMAL(18,10)"}

class DBConnectionFailed(Exception):
    pass

class MessageType(Enum):
    ADD_MODEL_EXECUTION_TIME = 1
    ADD_TEST_RESULT = 2
    ADD_TEST_ERROR = 3


class ResultsSender(object):
    def __init__(self, queue):
        self.queue = queue

    def addModelExecutionTime(self, fold, modelExecutionTime):
        self.queue.put([MessageType.ADD_MODEL_EXECUTION_TIME, fold, modelExecutionTime])

    def addTestResult(
        self, fold, docId, trueExpansion, predictedExpansion, testInstanceExecutionTime
    ):
        try:
            self.queue.put(
                [
                    MessageType.ADD_TEST_RESULT,
                    fold,
                    docId,
                    trueExpansion,
                    predictedExpansion,
                    testInstanceExecutionTime,
                ]
            )
        except Exception:
            logger.exception("Error sending message!")

    def addTestError(self, fold, docId, exceptionOcc=None):
        self.queue.put([MessageType.ADD_TEST_ERROR, fold, docId, ""])


class ResultsReportWritter(metaclass=ABCMeta):
    @abstractmethod
    def write_quality_results(self, fold, doc_id, quality_fields_per_article):
        """"""

    @abstractmethod
    def write_execution_times_results_per_article(
        self, fold, doc_id, exec_time_fields_per_article
    ):
        """"""

    @abstractmethod
    def write_final_results(self, fields):
        """"""

    def close(self):
        pass


class ResultsReportCSV(ResultsReportWritter):
    def __init__(
        self,
        experiment_name,
        experiment_parameters,
        report_name,
        save_results_per_article,
        general_report_fields,
        report_fields,
        quality_fields_per_article,
    ):
        self.report_name = report_name
        self.report_fieldnames = list(report_fields.keys())
        self.save_results_per_article = save_results_per_article

        self.general_report_fields = general_report_fields

        self.experiment_name = experiment_name
        self.experiment_parameters = experiment_parameters
        string_exp_param = "_".join(flatten_to_str_list(experiment_parameters))
        string_exp_param = string_exp_param.replace("/","-")
        if save_results_per_article:

            quality_results_filename = (
                f"quality_results_{experiment_name}_{string_exp_param}.csv"
            )
            exec_time_results_filename = (
                f"exec_time_results_{experiment_name}_{string_exp_param}.csv"
            )

            # to save execution times per article
            self.exec_time_results_file = open(
                FOLDER_LOGS + exec_time_results_filename, "w", encoding='utf-8-sig'
            )
            self.exec_time_results_file_writer = csv.DictWriter(
                self.exec_time_results_file,
                fieldnames=(
                    ["fold", "doc id"] + list(EXECUTION_TIME_PER_ARTICLES.keys())
                ),
            )
            self.exec_time_results_file_writer.writeheader()

            self.quality_results_file = open(
                FOLDER_LOGS + quality_results_filename, "w", encoding='utf-8-sig'
            )
            self.quality_results_file_writer = csv.DictWriter(
                self.quality_results_file,
                fieldnames=(
                    ["fold", "doc id"] + list(quality_fields_per_article.keys())
                ),
            )
            self.quality_results_file_writer.writeheader()

    def write_quality_results(self, fold, doc_id, quality_fields_per_article):
        if self.save_results_per_article:
            row_dict = {"fold": fold, "doc id": doc_id}
            row_dict.update(quality_fields_per_article)
            self.quality_results_file_writer.writerow(row_dict)

    def write_execution_times_results_per_article(
        self, fold, doc_id, exec_time_fields_per_article
    ):
        if self.save_results_per_article:
            row_dict = {
                "fold": fold,
                "doc id": doc_id,
                "Execution Times": exec_time_fields_per_article,
            }
            self.exec_time_results_file_writer.writerow(row_dict)

    def write_final_results(self, fields):
        """"""
        results_row_dict_fixed = {"experiment name": self.experiment_name}
        for k, v in self.experiment_parameters.items():
            if isinstance(v, str):
                str_v = v
            else:
                if isinstance(v, dict):
                    v.pop("train_data_manager_base")
                str_v = json.dumps(v)
            results_row_dict_fixed[k] = str_v

        with open(FOLDER_LOGS + self.report_name + ".csv", "a") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(results_row_dict_fixed.keys()) + self.report_fieldnames,
            )
            if f.tell() == 0:
                writer.writeheader()
            row_dict = {**results_row_dict_fixed, **fields}
            writer.writerow(row_dict)

    def close(self):
        if self.save_results_per_article:
            self.exec_time_results_file.close()
            self.quality_results_file.close()


def normalize_sql_field_name(orig_name):
    return orig_name.lower().replace(" ", "_")


class ResultsReportMySQL(ResultsReportWritter):

    FINAL_RESULTS_CREATE_TABLE = (
        "CREATE TABLE IF NOT EXISTS `{}` ("
        "  `experiment_no` int(11) NOT NULL AUTO_INCREMENT,"
        "  {}, "
        "  `node_name` varchar(25) NOT NULL,"
        "  PRIMARY KEY (`experiment_no`)"
        ") ENGINE=InnoDB"
    )

    SQL_INSERT_TEMPLATE = "INSERT INTO `{}` " "({}) " "VALUES ({})"

    BASE_REPORT_FIELDS = {
        "experiment_name": "VARCHAR(255)",
    }

    RESULTS_ARTICLE_CREATE_TABLE = (
        "CREATE TABLE IF NOT EXISTS `{}` ("
        "  {}, "
        "  `node_name` varchar(25) NOT NULL"
        ") ENGINE=InnoDB"
        " CHARACTER SET utf8mb4"
    )

    BASE_REPORT_ARTICLE_FIELDS = {"fold": "VARCHAR(10)", "doc_id": "VARCHAR(255)"}

    def __init__(
        self,
        experiment_name,
        experiment_parameters,
        report_name,
        save_results_per_article,
        general_report_fields,
        report_fields,
        quality_fields_per_article,
        db_config,
    ):

        self.db_config = db_config
        self.db_cnx = None

        self.report_name = report_name
        self.general_report_fields = general_report_fields
        self.report_fields = {
            **self.BASE_REPORT_FIELDS,
            **general_report_fields,
            **report_fields,
        }
        self.save_results_per_article = save_results_per_article

        self.experiment_name = experiment_name
        self.experiment_parameters = experiment_parameters

        self.quality_fields_per_article = {
            **self.BASE_REPORT_FIELDS,
            **general_report_fields,
            **self.BASE_REPORT_ARTICLE_FIELDS,
            **quality_fields_per_article,
        }

        self.execution_times_fields_per_article = {
            **self.BASE_REPORT_FIELDS,
            **general_report_fields,
            **self.BASE_REPORT_ARTICLE_FIELDS,
            **EXECUTION_TIME_PER_ARTICLES,
        }

        node_name = platform.node()

        self.results_row_dict_fixed = {
            "experiment name": self.experiment_name,
            "node_name": node_name,
        }
        for k, v in self.experiment_parameters.items():
            if v is None:
                str_v = ""
            elif isinstance(v, str):
                str_v = v
            else:
                str_v = json.dumps(v)
            self.results_row_dict_fixed[k] = str_v

        if self.save_results_per_article:
            self.create_per_article_tables()

    def format_fields_for_create_table(self, fields):
        fields_create_table = " ,".join(
            [
                f"`{normalize_sql_field_name(name)}` {type}"
                for name, type in fields.items()
            ]
        )
        return fields_create_table

    def get_connection(self):
        if self.db_cnx and self.db_cnx.is_connected():
            return self.db_cnx

        # Connect to MySQL Server
        try:
            cnx = mysql.connector.connect(**self.db_config)

            self.db_cnx = cnx
            return cnx
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logger.error("Something is wrong with your user name or password")
                raise err
            if err.errno == errorcode.ER_BAD_DB_ERROR:
                logger.error("Database does not exist")
                raise err
           
            logger.exception("MySQL DB connection error")
            raise DBConnectionFailed("MySQL DB connection error") from err
    
    def write_faield_statements_to_file(self, *statements_list):
        logger.exception("Failed to create tables in DB, saving statement to a file.")
        with open(FILE_FAILED_DB_STATEMENTS, 'a', encoding='utf-8-sig') as failed_stm_file:
            for statement in statements_list:
                failed_stm_file.write(statement + ";\n\n")
    
    def create_per_article_tables(self):
        self.table_name_quality_per_article = (
            "acronym_quality_per_article_" + self.report_name
        )
        fields_create_table = self.format_fields_for_create_table(
            self.quality_fields_per_article
        )
        create_table_quality = self.RESULTS_ARTICLE_CREATE_TABLE.format(
            self.table_name_quality_per_article, fields_create_table
        )

        self.table_name_exec_time__per_article = (
            "acronym_exec_time_per_article_" + self.report_name
        )
        fields_create_table = self.format_fields_for_create_table(
            self.execution_times_fields_per_article
        )
        create_table_exec_times = self.RESULTS_ARTICLE_CREATE_TABLE.format(
            self.table_name_exec_time__per_article, fields_create_table
        )

        try:
            cnx = self.get_connection()
            cursor = cnx.cursor()
            cursor.execute(create_table_quality)
            cursor.execute(create_table_exec_times)
            cursor.close()
            cnx.commit()
        except DBConnectionFailed:
            logger.exception("Failed to create tables in DB, saving statement to a file.")
            self.write_faield_statements_to_file(create_table_quality, create_table_exec_times)

    def write_quality_results(self, fold, doc_id, quality_fields_per_article):
        if self.save_results_per_article:
            fields = {
                **self.results_row_dict_fixed,
                "fold": fold,
                "doc_id": doc_id,
                **quality_fields_per_article,
            }
            add_results, values = self.get_insert_statement(
                self.table_name_quality_per_article, fields
            )

            try:
                cnx = self.get_connection()
    
                cursor = cnx.cursor()
                cursor.execute(add_results, values)
                cursor.close()
                cnx.commit()
            except DBConnectionFailed:
                logger.exception("Failed to send quality results to DB, saving statements to a file.")
                self.write_faield_statements_to_file(add_results % tuple(['"' + str(va) + '"' for va in values]))
                

    # TODO mudar exec var name
    def write_execution_times_results_per_article(
        self, fold, doc_id, exec_time_fields_per_article
    ):
        if self.save_results_per_article:
            fields = {
                **self.results_row_dict_fixed,
                "fold": fold,
                "doc_id": doc_id,
                "Execution Times": str(exec_time_fields_per_article),
            }
            add_results, values = self.get_insert_statement(
                self.table_name_exec_time__per_article, fields
            )
            try:
                cnx = self.get_connection()
    
                cursor = cnx.cursor()
                cursor.execute(add_results, values)
                cursor.close()
                cnx.commit()
            except DBConnectionFailed:
                logger.exception("Failed to send execution times results to DB, saving statements to a file.")
                self.write_faield_statements_to_file(add_results % tuple(['"' + str(va) + '"' for va in values]))

    def convert_type_to_python(self, val):
        converted_value = getattr(val, "tolist", lambda: val)()
        return converted_value

    def get_insert_statement(self, table_name, fields):
        row_dict = {normalize_sql_field_name(name): v for name, v in fields.items()}
        add_results = self.SQL_INSERT_TEMPLATE.format(
            table_name,
            ", ".join([f"`{name}`" for name in row_dict.keys()]),
            ", ".join(["%s"] * len(row_dict)),
        )
        values_list = [self.convert_type_to_python(v) for v in row_dict.values()]
        return add_results, values_list

    def write_final_results(self, fields):
        """"""

        table_name = "acronym_" + self.report_name
        fields_create_table = self.format_fields_for_create_table(self.report_fields)
        create_table = self.FINAL_RESULTS_CREATE_TABLE.format(
            table_name, fields_create_table
        )

        fields = {**self.results_row_dict_fixed, **fields}
        add_results, values = self.get_insert_statement(table_name, fields)
        try:
            cnx = self.get_connection()
    
            cursor = cnx.cursor()
            cursor.execute(create_table)
            cursor.execute(add_results, values)
            cursor.close()
            cnx.commit()
        except DBConnectionFailed:
            logger.error("Failed to send final results to DB, saving statements to a file.")
            self.write_faield_statements_to_file(create_table, add_results % tuple(['"' + str(va) + '"' for va in values]))
            
    def close(self):
        if self.db_cnx.is_connected:
            self.db_cnx.close()


class ResultsReporterBase(metaclass=ABCMeta):
    """
    classdocs
    """

    def __init__(
        self,
        experiment_name,
        experiment_parameters,
        report_name,
        save_results_per_article=False,
        db_config=None,
        cummulativeResultsInit={},
        general_report_fields={},
        quality_fields=None,
        quality_fields_per_article=None,
    ):
        """
        Constructor
        """
        self.experiment_name = experiment_name
        self.experiment_parameters = experiment_parameters

        self.save_results_per_article = save_results_per_article

        report_fields = {**BASE_FIELDS, **quality_fields, **EXECUTION_TIME_FIELDS}

        self.results_writters = []
        self.results_writters.append(
            ResultsReportCSV(
                experiment_name,
                experiment_parameters,
                report_name,
                save_results_per_article,
                general_report_fields,
                report_fields,
                quality_fields_per_article,
            )
        )

        if db_config:
            self.results_writters.append(
                ResultsReportMySQL(
                    experiment_name,
                    experiment_parameters,
                    report_name,
                    save_results_per_article,
                    general_report_fields,
                    report_fields,
                    quality_fields_per_article,
                    db_config,
                )
            )

        # Open file
        manager = Manager()
        self.queue = manager.JoinableQueue()
        self.t = Thread(target=self._worker)
        self.t.daemon = True
        self.t.start()

        self.cumulativeResults = {}
        self.articlesWithErrors = {}
        self.articlesTest = {}

        self.db_config = db_config

        self.cummulativeResultsInit = {
            "total_test_execution_time": ExecutionTimeObserver(),
            "model_execution_time": ExecutionTimeObserver(),
        }

        self.cummulativeResultsInit.update(cummulativeResultsInit)

    def __enter__(self):
        return ResultsSender(self.queue)

    def _worker(self):
        while True:
            args = self.queue.get()
            if args is None:
                self.queue.task_done()
                # self.queue.close()
                break

            if len(args) < 3:
                logger.error("Invalid message " + args)
                self.queue.task_done()
                continue

            messageType = args[0]
            args = args[1:]
            if messageType == MessageType.ADD_MODEL_EXECUTION_TIME:
                self._processModelExecutionTime(*args)
            elif messageType == MessageType.ADD_TEST_RESULT:
                self._processTestResult(*args)
            elif messageType == MessageType.ADD_TEST_ERROR:
                self._processTestError(*args)
            else:
                logger.error(
                    "Unkown message type: "
                    + str(messageType)
                    + " args: "
                    + ",".join(args)
                )

            # time.sleep(0.4)
            self.queue.task_done()

    def _processModelExecutionTime(self, fold, modelExecutionTime):
        foldCummulativeResults = self.cumulativeResults.setdefault(
            fold, copy.deepcopy(self.cummulativeResultsInit)
        )
        foldCummulativeResults["model_execution_time"] += modelExecutionTime

    def _processTestError(self, fold, docId, exceptionOcc):
        foldArticlesWithErrors = self.articlesWithErrors.setdefault(fold, [])
        foldArticlesWithErrors.append(docId)

    @abstractmethod
    def _process_quality_results(
        self,
        fold_cummulative_results,
        results_writer,
        fold,
        doc_id,
        actual_expansions,
        predicted_expansions,
    ):
        """"""

    def _processTestResult(
        self,
        fold,
        doc_id,
        actual_expansions,
        predicted_expansions,
        testInstanceExecutionTime,
    ):
        foldCummulativeResults = self.cumulativeResults.setdefault(
            fold, copy.deepcopy(self.cummulativeResultsInit)
        )

        self.articlesTest.setdefault(fold, set()).add(doc_id)
        foldCummulativeResults["total_test_execution_time"] += testInstanceExecutionTime

        if self.save_results_per_article:

            def results_writer(quality_fields_per_article):
                for writter in self.results_writters:
                    writter.write_quality_results(
                        fold, doc_id, quality_fields_per_article
                    )

            for writter in self.results_writters:
                writter.write_execution_times_results_per_article(
                    fold, doc_id, testInstanceExecutionTime
                )

        else:
            results_writer = None

        self._process_quality_results(
            foldCummulativeResults,
            results_writer,
            fold,
            doc_id,
            actual_expansions,
            predicted_expansions,
        )

    @abstractmethod
    def _compute_fold_quality_results(self, fold, fold_results):
        """"""

    def _computeFoldResults(self):
        for fold, foldResults in self.cumulativeResults.items():

            foldResults["len_test_articles"] = len(
                self.articlesTest.setdefault(fold, set())
            )
            foldResults["len_skipped_articles"] = len(
                self.articlesWithErrors.setdefault(fold, set())
            )

            self._compute_fold_quality_results(fold, foldResults)

            logger.info("Results for fold %s:", str(fold))
            for k, v in foldResults.items():
                if isinstance(v, str) or not isinstance(v, collections.Iterable):
                    logger.info("%s: %s", k, str(v))

    @abstractmethod
    def _plot_quality_stats(self):
        """"""

    def plotStats(self):

        testArticles = [
            item["len_test_articles"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        sumTestArticles = sum(testArticles)

        skippedArticles = [
            item["len_skipped_articles"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        sumSkippedArticles = sum(skippedArticles)

        logger.critical(
            "Processed %d rounds out of %d",
            len(testArticles),
            len(self.cumulativeResults.values()),
        )

        logger.critical(
            "Total articles: %d, skipped: %d", sumTestArticles, sumSkippedArticles
        )

        quality_fields = self._plot_quality_stats()

        avgModelExecutionTimes = mean(
            [
                item["model_execution_time"].getTime()
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        avgTestExecutionTimes = sum(
            [
                item["total_test_execution_time"].getTime()
                for item in self.cumulativeResults.values()
                if item != None
            ]
        ) / sum(
            [
                item["len_test_articles"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        logger.critical(
            "Execution Times, for model: %f, per article: %f",
            avgModelExecutionTimes,
            avgTestExecutionTimes,
        )

        base_fields_values = [
            len(testArticles),
            len(self.cumulativeResults.values()),
            sumTestArticles,
            sumSkippedArticles,
        ]
        exec_time_fields_values = [avgModelExecutionTimes, avgTestExecutionTimes]

        return {
            **dict(zip(BASE_FIELDS.keys(), base_fields_values)),
            **quality_fields,
            **dict(zip(EXECUTION_TIME_FIELDS.keys(), exec_time_fields_values)),
        }

    def extractScores(self, inputs):
        result = []
        for item in inputs:
            result.append(item[0])
        return result

    def _computeResults(self):
        self._computeFoldResults()

        metrics_dict = self.plotStats()

        for writter in self.results_writters:
            writter.write_final_results(metrics_dict)

    def __exit__(self, exc_type, exc_value, traceback):
        self.queue.join()
        self.queue.put(None)
        self.t.join()

        self._computeResults()

        for writter in self.results_writters:
            writter.close()
