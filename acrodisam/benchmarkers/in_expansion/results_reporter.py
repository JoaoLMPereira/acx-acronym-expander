"""
@author: jpereira
"""

from benchmarkers.results_reporter_base import ResultsReporterBase
from string_constants import REPORT_IN_EXPANSION_NAME
from helper import AcronymExpansion
from nltk.metrics import agreement
from nltk.metrics.distance import jaccard_distance, masi_distance
from numpy import mean
from DatasetParsers.expansion_linkage import _resolve_exp_acronym_db
from Logger import logging

logger = logging.getLogger(__name__)

GENERAL_REPORT_FIELDS = {
    "in_expander": "VARCHAR(255)",
    "in_expander_args": "VARCHAR(2000)",
}

SQL_FRACTION_TYPE = "DECIMAL(6, 5)"

QUALITY_FIELDS_LIST = [
    "Precision Acronyms In",
    "Recall Acronyms In",
    "F1 Acronyms In",
    "Alpha Jaccard Acronyms In",
    "Alpha MASI Acronyms In",
    "Kappa Jaccard Acronyms In",
    "Kappa MASI Acronyms In",
    "Precision Expansions In",
    "Recall Expansions In",
    "F1 Expansions In",
    "Alpha Jaccard Expansions In",
    "Alpha MASI Expansions In",
    "Kappa Jaccard Expansions In",
    "Kappa MASI Expansions In",
    "Precision Pairs In",
    "Recall Pairs In",
    "F1 Pairs In",
    "Alpha Jaccard Pairs In",
    "Alpha MASI Pairs In",
    "Kappa Jaccard Pairs In",
    "Kappa MASI Pairs In",
    "Dict Precision Acronyms In",
    "Dict Recall Acronyms In",
    "Dict F1 Acronyms In",
    "Dict Precision Pairs In",
    "Dict Recall Pairs In",
    "Dict F1 Pairs In",
]

QUALITY_FIELDS = {field_name: SQL_FRACTION_TYPE for field_name in QUALITY_FIELDS_LIST}

QUALITY_FIELDS_PER_ARTICLE = {
    "acronym": "VARCHAR(20)",
    "actual_expansion": "VARCHAR(255)",
    "predicted_expansion": "VARCHAR(255)",
    "success": "BOOLEAN",
}


class ResultsReporter(ResultsReporterBase):
    """Reports all of the metrics for an in expansion benchmark execution.
    When initializing an in expansion benchmark this class should be used as the result reporter.
    For reference the _process_quality_results method is called one time per article, the _compute_fold_quality_results
    one time per fold and the _plot_quality_stats at the end of the benchmark (i.e. after all the folds are processed).
    """

    def __init__(
        self,
        experiment_name: str,
        experiment_parameters,
        report_name: str = REPORT_IN_EXPANSION_NAME,
        save_results_per_article: bool = False,
        db_config=None,
    ):
        """Initializes the results reporter.
        Args:
            experiment_name (str):
             the name of the experience
            experiment_parameters (dict):
             a dictionary with the where the keys are the name of the parameter and the values are the value of the parameter.
             Example: {'out_expander': 'cossim', 'out_expander_args': 'classic_context_vector'}
            report_name (str, optional):
             the name that should be given to the report. Defaults to REPORT_IN_EXPANSION_NAME.
            save_results_per_article (bool, optional):
             a boolean that indicates if we should save results per article. Defaults to False.
            db_config (dict, optional):
             A dictionary with the configuration for the database.
             Example: {'host': 'example.db.com', 'port': 3306, 'database': 'example', 'user': 'Bob', 'password': 'xxxx'}. Defaults to None.
        """

        cummulative_results_init = {
            "acronyms_in": {"correct": 0, "extracted": 0, "gold": 0},
            "expansions_in": {"correct": 0, "extracted": 0, "gold": 0},
            "pairs_in": {"correct": 0, "extracted": 0, "gold": 0},
            "annotations_data_acro_in": [],
            "annotations_data_exp_in": [],
            "annotations_data_pairs_in": [],
            "gold_acronym_db": {},
            "predicted_acronym_db": {},
        }

        super().__init__(
            experiment_name,
            experiment_parameters,
            report_name,
            save_results_per_article,
            db_config,
            cummulative_results_init,
            GENERAL_REPORT_FIELDS,
            QUALITY_FIELDS,
            QUALITY_FIELDS_PER_ARTICLE,
        )

    def _register_quality_results(
        self,
        fold_cummulative_results,
        results_writer,
        doc_id,
        actual_expansions,
        predicted_expansions,
    ):
        """Compares the predicted expansions by the system against the actual expansions for each test document.
        Registers correct, extracted and gold acronyms,expansions and acronym-expansions pairs by the system for each test document.
        Additionaly it also registers the expansions for each acronym.
        Args:
            fold_cummulative_results (dict):
             a dictionary where the key is the name of a measure and the value is the cumulative value, these values are used to compute the final measures.
            results_writer (function):
             the results writer to be used
            doc_id (int):
             the document id of the article currently being analysed
            actual_expansions (dict):
             a dictionary where each key is an acronym and each value is an expansion
            predicted_expansions (dict):
             a dicionary where each key is an acronym and each value is a list that holds the expansion and the extraction module used
        """

        for acronym, actual_expansion in actual_expansions.items():

            if acronym in fold_cummulative_results["gold_acronym_db"].keys():
                fold_cummulative_results["gold_acronym_db"][acronym].append(
                    (actual_expansion, doc_id)
                )
            else:
                fold_cummulative_results["gold_acronym_db"][acronym] = [
                    (actual_expansion, doc_id)
                ]

            fold_cummulative_results["acronyms_in"]["gold"] += 1
            fold_cummulative_results["expansions_in"]["gold"] += 1
            fold_cummulative_results["pairs_in"]["gold"] += 1

            predicted_expansion = predicted_expansions.pop(acronym, None)

            # try removing s or add s
            if predicted_expansion is None and len(acronym) > 1 and acronym[-1] == "s":
                predicted_expansion = predicted_expansions.pop(acronym[:-1], None)
            elif (
                predicted_expansion is None and len(acronym) > 1 and acronym[-1] != "s"
            ):
                predicted_expansion = predicted_expansions.pop(acronym + "s", None)

            if predicted_expansion:

                if acronym in fold_cummulative_results["predicted_acronym_db"].keys():
                    fold_cummulative_results["predicted_acronym_db"][acronym].append(
                        (predicted_expansion, doc_id)
                    )
                else:
                    fold_cummulative_results["predicted_acronym_db"][acronym] = [
                        (predicted_expansion, doc_id)
                    ]

                fold_cummulative_results["acronyms_in"]["extracted"] += 1
                fold_cummulative_results["expansions_in"]["extracted"] += 1
                fold_cummulative_results["pairs_in"]["extracted"] += 1

                fold_cummulative_results["acronyms_in"]["correct"] += 1

                if AcronymExpansion.areExpansionsSimilar(
                    actual_expansion.strip().lower(),
                    predicted_expansion.strip().lower(),
                ):

                    fold_cummulative_results["expansions_in"]["correct"] += 1
                    fold_cummulative_results["pairs_in"]["correct"] += 1

                    logger.debug(
                        "Expansion matching succeeded in doc_id %s, (%s): %s, %s"
                        % (doc_id, acronym, actual_expansion, predicted_expansion)
                    )

                    if results_writer:

                        results_writer(
                            dict(
                                zip(
                                    QUALITY_FIELDS_PER_ARTICLE,
                                    [
                                        acronym,
                                        actual_expansion,
                                        predicted_expansion,
                                        True,
                                    ],
                                )
                            )
                        )

                else:

                    logger.debug(
                        "Expansion matching failed in doc_id %s, (%s): %s, %s"
                        % (doc_id, acronym, actual_expansion, predicted_expansion)
                    )

                    if results_writer:

                        results_writer(
                            dict(
                                zip(
                                    QUALITY_FIELDS_PER_ARTICLE,
                                    [
                                        acronym,
                                        actual_expansion,
                                        predicted_expansion,
                                        False,
                                    ],
                                )
                            )
                        )
            else:
                logger.debug(
                    "Expansion matching failed in doc_id %s, (%s): %s, %s"
                    % (doc_id, acronym, actual_expansion, None)
                )

                if results_writer:

                    results_writer(
                        dict(
                            zip(
                                QUALITY_FIELDS_PER_ARTICLE,
                                [
                                    acronym,
                                    actual_expansion,
                                    "",
                                    False,
                                ],
                            )
                        )
                    )
        for acronym, expansion in predicted_expansions.items():

            if acronym in fold_cummulative_results["predicted_acronym_db"].keys():
                fold_cummulative_results["predicted_acronym_db"][acronym].append(
                    (expansion, doc_id)
                )
            else:
                fold_cummulative_results["predicted_acronym_db"][acronym] = [
                    (expansion, doc_id)
                ]

            fold_cummulative_results["acronyms_in"]["extracted"] += 1
            fold_cummulative_results["pairs_in"]["extracted"] += 1

            logger.debug(
                "Expansion matching failed in doc_id %s, (%s): %s, %s"
                % (doc_id, acronym, None, expansion)
            )

            if results_writer:

                results_writer(
                    dict(
                        zip(
                            QUALITY_FIELDS_PER_ARTICLE,
                            [
                                acronym,
                                "",
                                expansion,
                                False,
                            ],
                        )
                    )
                )

    def _process_quality_results(
        self,
        fold_cummulative_results,
        results_writer,
        fold,
        doc_id,
        actual_expansions,
        predicted_expansions,
    ):
        """Processes an article and registers correct/extracted/gold acronyms,expansions and acronym-expansions pairs by the system for each test document.
        Data structures are also created in order to calculate Cohen's kappa and Krippendorff's alpha.
        Args:
            fold_cummulative_results (dict):
             a dictionary where all the necessary data about correct and incorrect predictions should be stored.
            results_writer (function):
             the results writer to be used
            doc_id (int):
             the document id of the article currently being analysed
            actual_expansions (dict):
             a dictionary where each key is an acronym and each value is a list that holds the expansion and a boolean indicating if the expansion is present in the article
            predicted_expansions (dict):
             a dicionary where each key is an acronym and each value is a list that holds the expansion and the extraction module used
        """

        actual_expansions_in = {}

        for k, v in actual_expansions.items():
            # it's a dataset with both in expansions and out expansions displayed by a flag
            if v != None and len(v) == 2:
                if v[1]:
                    actual_expansions_in[k.lower()] = v[0]
            else:
                if v != None:
                    actual_expansions_in[k.lower()] = v

        predicted_expansions_in = {
            k.lower(): v for k, v in predicted_expansions.items()
        }

        if len(actual_expansions_in) != 0 and len(predicted_expansions_in) != 0:
            fold_cummulative_results["annotations_data_acro_in"].append(
                (("u1"), str(doc_id), frozenset(actual_expansions_in.keys()))
            )
            fold_cummulative_results["annotations_data_acro_in"].append(
                (("u2"), str(doc_id), frozenset(predicted_expansions_in.keys()))
            )
            fold_cummulative_results["annotations_data_exp_in"].append(
                (("u1"), str(doc_id), frozenset(actual_expansions_in.values()))
            )
            fold_cummulative_results["annotations_data_exp_in"].append(
                (("u2"), str(doc_id), frozenset(predicted_expansions_in.values()))
            )
            fold_cummulative_results["annotations_data_pairs_in"].append(
                (("u1"), str(doc_id), frozenset(actual_expansions_in.items()))
            )
            fold_cummulative_results["annotations_data_pairs_in"].append(
                (("u2"), str(doc_id), frozenset(predicted_expansions_in.items()))
            )

        self._register_quality_results(
            fold_cummulative_results,
            results_writer,
            doc_id,
            actual_expansions_in,
            predicted_expansions_in,
        )

    def _calculate_dict_level_results(
        self, gold_acronym_db, predicted_acronym_db, fold_results
    ):
        """Calculates dictionary level performance, i.e. precision, recall, f1 considering only unique acronyms and pairs.
        Args:
            gold_acronym_db (dict):
             a dictionary where each key is an acronym and each value is a list with tuples of the form (expansion, article_id). This dictionary
             is the acronym db for the dataset being tested.
            predicted_acronym_db (dict): a dictionary where each key is an acronym and each value is a list with tuples of the form (expansion, article_id). This dictionary
             is the acronym db for the predictions made on the test dataset.
            fold_results (dict):
             a dictionary where all the metrics are stored.
        """

        gold_acronym_db_clean = {}

        # substituting similar expansions with the same expansion
        _resolve_exp_acronym_db(gold_acronym_db, gold_acronym_db_clean)

        predicted_acronym_db_clean = {}

        _resolve_exp_acronym_db(predicted_acronym_db, predicted_acronym_db_clean)

        # removing duplicate expansions
        for acro, expansions in gold_acronym_db_clean.items():
            expansions_in = []
            for expansion in expansions:
                expansions_in.append(expansion[0])

            expansions_in = list(dict.fromkeys(expansions_in))
            gold_acronym_db_clean[acro] = expansions_in

        for acro, expansions in predicted_acronym_db_clean.items():
            expansions_in = []
            for expansion in expansions:
                expansions_in.append(expansion[0])

            expansions_in = list(dict.fromkeys(expansions_in))
            predicted_acronym_db_clean[acro] = expansions_in

        results = {
            "unique_acronyms": {"correct": 0, "extracted": 0, "gold": 0},
            "unique_pairs": {"correct": 0, "extracted": 0, "gold": 0},
        }

        for gold_acro, gold_exps in gold_acronym_db_clean.items():
            results["unique_acronyms"]["gold"] += len(gold_exps)
            results["unique_pairs"]["gold"] += len(gold_exps)

            predicted_expansions = predicted_acronym_db_clean.pop(gold_acro, None)
            if (
                predicted_expansions is None
                and len(gold_acro) > 1
                and gold_acro[-1] == "s"
            ):
                predicted_expansions = predicted_acronym_db_clean.pop(
                    gold_acro[:-1], None
                )
            elif (
                predicted_expansions is None
                and len(gold_acro) > 1
                and gold_acro[-1] != "s"
            ):
                predicted_expansions = predicted_acronym_db_clean.pop(
                    gold_acro + "s", None
                )

            if predicted_expansions != None:
                results["unique_acronyms"]["extracted"] += len(predicted_expansions)
                results["unique_pairs"]["extracted"] += len(predicted_expansions)
                if len(gold_exps) - len(predicted_expansions) > 0:
                    results["unique_acronyms"]["correct"] += len(predicted_expansions)
                else:
                    results["unique_acronyms"]["correct"] += len(gold_exps)

                for pred_exp in predicted_expansions:
                    for exp in gold_exps:
                        if AcronymExpansion.areExpansionsSimilar(pred_exp, exp):
                            results["unique_pairs"]["correct"] += 1
                            break

        for acro, exp in predicted_acronym_db_clean.items():
            results["unique_acronyms"]["extracted"] += len(exp)
            results["unique_pairs"]["extracted"] += len(exp)

        try:
            fold_results["dict_precision_acronym"] = (
                results["unique_acronyms"]["correct"]
                / results["unique_acronyms"]["extracted"]
            )
        except ZeroDivisionError:
            fold_results["dict_precision_acronym"] = 0

        try:
            fold_results["dict_recall_acronym"] = (
                results["unique_acronyms"]["correct"]
                / results["unique_acronyms"]["gold"]
            )

        except ZeroDivisionError:
            fold_results["dict_recall_acronym"] = 0

        try:
            fold_results["dict_f1_score_acronym"] = 2 * (
                (
                    fold_results["dict_precision_acronym"]
                    * fold_results["dict_recall_acronym"]
                )
                / (
                    fold_results["dict_precision_acronym"]
                    + fold_results["dict_recall_acronym"]
                )
            )
        except ZeroDivisionError:
            fold_results["dict_f1_score_acronym"] = 0

        try:
            fold_results["dict_precision_pair"] = (
                results["unique_pairs"]["correct"]
                / results["unique_pairs"]["extracted"]
            )
        except ZeroDivisionError:
            fold_results["dict_precision_pair"] = 0

        try:
            fold_results["dict_recall_pair"] = (
                results["unique_pairs"]["correct"] / results["unique_pairs"]["gold"]
            )
        except ZeroDivisionError:
            fold_results["dict_recall_pair"] = 0

        try:
            fold_results["dict_f1_score_pair"] = 2 * (
                (fold_results["dict_precision_pair"] * fold_results["dict_recall_pair"])
                / (
                    fold_results["dict_precision_pair"]
                    + fold_results["dict_recall_pair"]
                )
            )
        except ZeroDivisionError:
            fold_results["dict_f1_score_pair"] = 0

    def _compute_fold_quality_results(self, fold, fold_results):
        """Calculates all final metrics for the current fold based on the cumulative results.
        Args:
            fold (str):
             the fold for which metrics are going to be calculated
            fold_results (dict):
             a dictionary where all the metrics are stored.
        """

        try:
            fold_results["precision_acronyms_in"] = (
                fold_results["acronyms_in"]["correct"]
                / fold_results["acronyms_in"]["extracted"]
            )
        except ZeroDivisionError:
            fold_results["precision_acronyms_in"] = 0

        try:
            fold_results["recall_acronyms_in"] = (
                fold_results["acronyms_in"]["correct"]
                / fold_results["acronyms_in"]["gold"]
            )
        except ZeroDivisionError:
            fold_results["recall_acronyms_in"] = 0
        try:
            fold_results["f1_score_acronyms_in"] = 2 * (
                (
                    fold_results["precision_acronyms_in"]
                    * fold_results["recall_acronyms_in"]
                )
                / (
                    fold_results["precision_acronyms_in"]
                    + fold_results["recall_acronyms_in"]
                )
            )
        except ZeroDivisionError:
            fold_results["f1_score_acronyms_in"] = 0

        try:
            fold_results["precision_expansions_in"] = (
                fold_results["expansions_in"]["correct"]
                / fold_results["expansions_in"]["extracted"]
            )
        except ZeroDivisionError:
            fold_results["precision_expansions_in"] = 0

        try:
            fold_results["recall_expansions_in"] = (
                fold_results["expansions_in"]["correct"]
                / fold_results["expansions_in"]["gold"]
            )
        except ZeroDivisionError:
            fold_results["recall_expansions_in"] = 0

        try:
            fold_results["f1_score_expansions_in"] = 2 * (
                (
                    fold_results["precision_expansions_in"]
                    * fold_results["recall_expansions_in"]
                )
                / (
                    fold_results["precision_expansions_in"]
                    + fold_results["recall_expansions_in"]
                )
            )
        except ZeroDivisionError:
            fold_results["f1_score_expansions_in"] = 0

        try:
            fold_results["precision_pairs_in"] = (
                fold_results["pairs_in"]["correct"]
                / fold_results["pairs_in"]["extracted"]
            )
        except ZeroDivisionError:
            fold_results["precision_pairs_in"] = 0

        try:
            fold_results["recall_pairs_in"] = (
                fold_results["pairs_in"]["correct"] / fold_results["pairs_in"]["gold"]
            )
        except ZeroDivisionError:
            fold_results["recall_pairs_in"] = 0

        try:
            fold_results["f1_score_pairs_in"] = 2 * (
                (fold_results["precision_pairs_in"] * fold_results["recall_pairs_in"])
                / (fold_results["precision_pairs_in"] + fold_results["recall_pairs_in"])
            )
        except ZeroDivisionError:
            fold_results["f1_score_pairs_in"] = 0

        self._calculate_dict_level_results(
            fold_results["gold_acronym_db"],
            fold_results["predicted_acronym_db"],
            fold_results,
        )

        task_acro_in = agreement.AnnotationTask(
            data=fold_results["annotations_data_acro_in"],
            distance=jaccard_distance,
        )

        task_exp_in = agreement.AnnotationTask(
            data=fold_results["annotations_data_exp_in"],
            distance=jaccard_distance,
        )

        task_pairs_in = agreement.AnnotationTask(
            data=fold_results["annotations_data_pairs_in"],
            distance=jaccard_distance,
        )

        if len(fold_results["annotations_data_acro_in"]) != 0:
            fold_results["alpha_jaccard_acro_in"] = task_acro_in.alpha()
            fold_results["kappa_jaccard_acro_in"] = task_acro_in.kappa()
        else:
            fold_results["alpha_jaccard_acro_in"] = 0
            fold_results["kappa_jaccard_acro_in"] = 0

        if len(fold_results["annotations_data_exp_in"]) != 0:
            fold_results["alpha_jaccard_exp_in"] = task_exp_in.alpha()
            fold_results["kappa_jaccard_exp_in"] = task_exp_in.kappa()
        else:
            fold_results["alpha_jaccard_exp_in"] = 0
            fold_results["kappa_jaccard_exp_in"] = 0

        if len(fold_results["annotations_data_pairs_in"]) != 0:
            fold_results["alpha_jaccard_pairs_in"] = task_pairs_in.alpha()
            fold_results["kappa_jaccard_pairs_in"] = task_pairs_in.kappa()
        else:
            fold_results["alpha_jaccard_pairs_in"] = 0
            fold_results["kappa_jaccard_pairs_in"] = 0

        task_acro_in = agreement.AnnotationTask(
            data=fold_results["annotations_data_acro_in"],
            distance=masi_distance,
        )

        task_exp_in = agreement.AnnotationTask(
            data=fold_results["annotations_data_exp_in"],
            distance=masi_distance,
        )

        task_pairs_in = agreement.AnnotationTask(
            data=fold_results["annotations_data_pairs_in"],
            distance=masi_distance,
        )

        if len(fold_results["annotations_data_acro_in"]) != 0:
            fold_results["alpha_masi_acro_in"] = task_acro_in.alpha()
            fold_results["kappa_masi_acro_in"] = task_acro_in.kappa()
        else:
            fold_results["alpha_masi_acro_in"] = 0
            fold_results["kappa_masi_acro_in"] = 0

        if len(fold_results["annotations_data_exp_in"]) != 0:
            fold_results["alpha_masi_exp_in"] = task_exp_in.alpha()
            fold_results["kappa_masi_exp_in"] = task_exp_in.kappa()
        else:
            fold_results["alpha_masi_exp_in"] = 0
            fold_results["kappa_masi_exp_in"] = 0

        if len(fold_results["annotations_data_pairs_in"]) != 0:
            fold_results["alpha_masi_pairs_in"] = task_pairs_in.alpha()
            fold_results["kappa_masi_pairs_in"] = task_pairs_in.kappa()
        else:
            fold_results["alpha_masi_pairs_in"] = 0
            fold_results["kappa_masi_pairs_in"] = 0

    def _plot_quality_stats(self):
        """Calculates the final metrics for the benchmark taking into account all of the folds
        Returns:
            list: a list with all of the metrics that were calculated
        """

        precision_acronyms_in = [
            item["precision_acronyms_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_acronyms_in = mean(precision_acronyms_in)

        recall_acronyms_in = [
            item["recall_acronyms_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_acronyms_in = mean(recall_acronyms_in)

        f1_score_acronyms_in = [
            item["f1_score_acronyms_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_acronyms_in = mean(f1_score_acronyms_in)

        logger.critical(
            "Acronyms in text -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_precision_acronyms_in,
            mean_recall_acronyms_in,
            mean_f1_score_acronyms_in,
        )

        precision_expansions_in = [
            item["precision_expansions_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_expansions_in = mean(precision_expansions_in)

        recall_expansions_in = [
            item["recall_expansions_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_expansions_in = mean(recall_expansions_in)

        f1_score_expansions_in = [
            item["f1_score_expansions_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_expansions_in = mean(f1_score_expansions_in)

        logger.critical(
            "Expansions in text -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_precision_expansions_in,
            mean_recall_expansions_in,
            mean_f1_score_expansions_in,
        )

        precision_pairs_in = [
            item["precision_pairs_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_pairs_in = mean(precision_pairs_in)

        recall_pairs_in = [
            item["recall_pairs_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_pairs_in = mean(recall_pairs_in)

        f1_score_pairs_in = [
            item["f1_score_pairs_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_pairs_in = mean(f1_score_pairs_in)

        logger.critical(
            "Pairs in text -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_precision_pairs_in,
            mean_recall_pairs_in,
            mean_f1_score_pairs_in,
        )

        dict_precision_acronyms_in = [
            item["dict_precision_acronym"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_dict_precision_acronyms_in = mean(dict_precision_acronyms_in)

        dict_recall_acronyms_in = [
            item["dict_recall_acronym"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_dict_recall_acronyms_in = mean(dict_recall_acronyms_in)

        dict_f1_score_acronyms_in = [
            item["dict_f1_score_acronym"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_dict_f1_score_acronyms_in = mean(dict_f1_score_acronyms_in)

        logger.critical(
            "Dict Acronyms -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_dict_precision_acronyms_in,
            mean_dict_recall_acronyms_in,
            mean_dict_f1_score_acronyms_in,
        )

        dict_precision_pairs_in = [
            item["dict_precision_pair"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_dict_precision_pairs_in = mean(dict_precision_pairs_in)

        dict_recall_pairs_in = [
            item["dict_recall_pair"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_dict_recall_pairs_in = mean(dict_recall_pairs_in)

        dict_f1_score_pairs_in = [
            item["dict_f1_score_pair"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_dict_f1_score_pairs_in = mean(dict_f1_score_pairs_in)

        logger.critical(
            "Dict Pairs -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_dict_precision_pairs_in,
            mean_dict_recall_pairs_in,
            mean_dict_f1_score_pairs_in,
        )

        alpha_jaccard_acro_in = [
            item["alpha_jaccard_acro_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_jaccard_exp_in = [
            item["alpha_jaccard_exp_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_jaccard_pairs_in = [
            item["alpha_jaccard_pairs_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_acro_in = [
            item["kappa_jaccard_acro_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_exp_in = [
            item["kappa_jaccard_exp_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_pairs_in = [
            item["kappa_jaccard_pairs_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        alpha_masi_acro_in = [
            item["alpha_masi_acro_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_masi_exp_in = [
            item["alpha_masi_exp_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_masi_pairs_in = [
            item["alpha_masi_pairs_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_acro_in = [
            item["kappa_masi_acro_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_exp_in = [
            item["kappa_masi_exp_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_pairs_in = [
            item["kappa_masi_pairs_in"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        measure_values = [
            mean_precision_acronyms_in,
            mean_recall_acronyms_in,
            mean_f1_score_acronyms_in,
            mean(alpha_jaccard_acro_in),
            mean(alpha_masi_acro_in),
            mean(kappa_jaccard_acro_in),
            mean(kappa_masi_acro_in),
            mean_precision_expansions_in,
            mean_recall_expansions_in,
            mean_f1_score_expansions_in,
            mean(alpha_jaccard_exp_in),
            mean(alpha_masi_exp_in),
            mean(kappa_jaccard_exp_in),
            mean(kappa_masi_exp_in),
            mean_precision_pairs_in,
            mean_recall_pairs_in,
            mean_f1_score_pairs_in,
            mean(alpha_jaccard_pairs_in),
            mean(alpha_masi_pairs_in),
            mean(kappa_jaccard_pairs_in),
            mean(kappa_masi_pairs_in),
            mean_dict_precision_acronyms_in,
            mean_dict_recall_acronyms_in,
            mean_dict_f1_score_acronyms_in,
            mean_dict_precision_pairs_in,
            mean_dict_recall_pairs_in,
            mean_dict_f1_score_pairs_in,
        ]

        return dict(zip(QUALITY_FIELDS_LIST, measure_values))
