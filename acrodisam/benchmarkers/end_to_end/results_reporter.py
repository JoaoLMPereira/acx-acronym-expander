"""Contains the results reporter class that should be used by all end to end benchmarks,

Created on Jun, 2021

@author: JRCasanova
"""
from collections import defaultdict
import collections
import itertools

from benchmarkers.results_reporter_base import ResultsReporterBase
from expansion_module import ExpansionModuleEnum
from helper import AcronymExpansion
from Logger import logging
from nltk.metrics import agreement
from nltk.metrics.distance import edit_distance, jaccard_distance, masi_distance
from numpy import mean, std
from string_constants import REPORT_END_TO_END_NAME
import benchmarkers.in_expansion.results_reporter
import benchmarkers.out_expansion.results_reporter


logger = logging.getLogger(__name__)

GENERAL_REPORT_FIELDS = {
    **benchmarkers.in_expansion.results_reporter.GENERAL_REPORT_FIELDS,
    **benchmarkers.out_expansion.results_reporter.GENERAL_REPORT_FIELDS,
    "follow_links": "BOOLEAN",
}

SQL_FRACTION_TYPE = "DECIMAL(6, 5)"

QUALITY_FIELDS_LIST = [
    "Precision Total Acronyms",
    "Recall Total Acronyms",
    "F1 Total Acronyms",
    "Alpha Jaccard Total Acronyms",
    "Alpha MASI Total Acronyms",
    "Kappa Jaccard Total Acronyms",
    "Kappa MASI Total Acronyms",
    "Precision Total Expansions",
    "Recall Total Expansions",
    "F1 Total Expansions",
    "Alpha Jaccard Total Expansions",
    "Alpha MASI Total Expansions",
    "Kappa Jaccard Total Expansions",
    "Kappa MASI Total Expansions",
    "Precision Total Pairs",
    "Recall Total Pairs",
    "F1 Total Pairs",
    "Micro Precision Total Pairs",
    "Micro Recall Total Pairs",
    "Micro F1 Total Pairs",
    "Alpha Jaccard Total Pairs",
    "Alpha MASI Total Pairs",
    "Kappa Jaccard Total Pairs",
    "Kappa MASI Total Pairs",
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
    "Micro Precision Pairs In",
    "Micro Recall Pairs In",
    "Micro F1 Pairs In",
    "Alpha Jaccard Pairs In",
    "Alpha MASI Pairs In",
    "Kappa Jaccard Pairs In",
    "Kappa MASI Pairs In",
    "Precision Acronyms Out",
    "Recall Acronyms Out",
    "F1 Acronyms Out",
    "Alpha Jaccard Acronyms Out",
    "Alpha MASI Acronyms Out",
    "Kappa Jaccard Acronyms Out",
    "Kappa MASI Acronyms Out",
    "Precision Expansions Out",
    "Recall Expansions Out",
    "F1 Expansions Out",
    "Alpha Jaccard Expansions Out",
    "Alpha MASI Expansions Out",
    "Kappa Jaccard Expansions Out",
    "Kappa MASI Expansions Out",
    "Precision Pairs Out",
    "Recall Pairs Out",
    "F1 Pairs Out",
    "Micro Precision Pairs Out",
    "Micro Recall Pairs Out",
    "Micro F1 Pairs Out",
    "Alpha Jaccard Pairs Out",
    "Alpha MASI Pairs Out",
    "Kappa Jaccard Pairs Out",
    "Kappa MASI Pairs Out",
    
    "Precision Acronyms Links",
    "Recall Acronyms Links",
    "F1 Acronyms Links",
    
    "Precision Expansions Links",
    "Recall Expansions Links",
    "F1 Expansions Links",
    
    "Precision Pairs Links",
    "Recall Pairs Links",
    "F1 Pairs Links",
    
    "Precision Acronyms Out in DB",
    "Recall Acronyms Out in DB",
    "F1 Acronyms Out in DB",
    
    "Precision Expansions Out in DB",
    "Recall Expansions Out in DB",
    "F1 Expansions Out in DB",
    
    "Precision Pairs Out in DB",
    "Recall Pairs Out in DB",
    "F1 Pairs Out in DB",
    
    "Precision Total Acronyms in DB",
    "Recall Total Acronyms in DB",
    "F1 Total Acronyms in DB",
    
    "Precision Total Expansions in DB",
    "Recall Total Expansions in DB",
    "F1 Total Expansions in DB",
    
    "Precision Total Pairs in DB",
    "Recall Total Pairs in DB",
    "F1 Total Pairs in DB",
]

QUALITY_FIELDS = {**{field_name: SQL_FRACTION_TYPE for field_name in QUALITY_FIELDS_LIST}, "Total Links Followed" : "INT(11)"}

QUALITY_FIELDS_PER_ARTICLE = {
    "acronym": "VARCHAR(20)",
    "actual_expansion": "VARCHAR(255)",
    "predicted_expansion": "VARCHAR(255)",
    "success": "BOOLEAN",
    "in_text": "BOOLEAN",
    "module_used": "VARCHAR(255)",
}


class ResultsReporter(ResultsReporterBase):
    """Reports all of the metrics for an end to end benchmark execution.

    When initializing an end to end benchmark this class should be used as the result reporter.

    For reference the _process_quality_results method is called one time per article, the _compute_fold_quality_results
    one time per fold and the _plot_quality_stats at the end of the benchmark (i.e. after all the folds are processed).
    """

    def __init__(
        self,
        experiment_name: str,
        experiment_parameters,
        report_name: str = REPORT_END_TO_END_NAME,
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
             the name that should be given to the report. Defaults to REPORT_END_TO_END_NAME.
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
            "acronyms_out": {"correct": 0, "extracted": 0, "gold": 0},
            "expansions_out": {"correct": 0, "extracted": 0, "gold": 0},
            "pairs_out": {"correct": 0, "extracted": 0, "gold": 0},
            "acronyms_links": {"correct": 0, "extracted": 0, "gold": 0},
            "expansions_links": {"correct": 0, "extracted": 0, "gold": 0},
            "pairs_links": {"correct": 0, "extracted": 0, "gold": 0},
            "acronyms_total": {"correct": 0, "extracted": 0, "gold": 0},
            "expansions_total": {"correct": 0, "extracted": 0, "gold": 0},
            "pairs_total": {"correct": 0, "extracted": 0, "gold": 0},
            "annotations_data_acro_in": [],
            "annotations_data_exp_in": [],
            "annotations_data_pairs_in": [],
            "annotations_data_acro_out": [],
            "annotations_data_exp_out": [],
            "annotations_data_pairs_out": [],
            "annotations_data_acro_total": [],
            "annotations_data_exp_total": [],
            "annotations_data_pairs_total": [],
            "total_links_followed": 0,
            "extracted_from_links": 0,
            "correct_from_links": 0,
            "incorrect_missing_label": 0
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

    def _acroexp_module_to_metric_suffix(self, module):
        if module == ExpansionModuleEnum.in_expander:
            return "in"
        elif module == ExpansionModuleEnum.link_follower:
            return "links"
        elif module == ExpansionModuleEnum.out_expander:
            return "out"
        
        return None
           
    def _simExpansionExists(self, testExp, expList):
        for candidateExp in expList:
            if AcronymExpansion.areExpansionsSimilar(testExp, candidateExp):
                return True

        return False 

    def _calculate_quality_results(
        self,
        fold_cummulative_results,
        results_writer,
        doc_id,
        actual_expansions,
        predicted_expansions,
        links_followed
    ):
        """Compares the predicted expansions by the system against the actual expansions for each test document.

        Registers correct, extracted and gold acronyms,expansions and acronym-expansions pairs by the system for each test document.

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
            in_text (bool):
             a boolean that indicates if we are processing pairs that have the expansion in text or not.
        """
        for acronym, actual_expansion_tuple in actual_expansions.items():
            actual_expansion = actual_expansion_tuple[0]
            in_text = actual_expansion_tuple[1]
            
            if in_text:
                in_or_out_text = "in"
            else:
                in_or_out_text = "out"
            fold_cummulative_results["acronyms_" + in_or_out_text]["gold"] += 1
            # TODO late consider to use expansion metrics for only when we get the acronym right. 
            #Maybe take acronyms identified with no expansion found
            fold_cummulative_results["expansions_" + in_or_out_text]["gold"] += 1
            fold_cummulative_results["pairs_" + in_or_out_text]["gold"] += 1
            
            fold_cummulative_results["acronyms_total"]["gold"] += 1
            fold_cummulative_results["expansions_total"]["gold"] += 1
            fold_cummulative_results["pairs_total"]["gold"] += 1

            if acronym in links_followed:
                fold_cummulative_results["acronyms_links"]["gold"] += 1
                fold_cummulative_results["expansions_links"]["gold"] += 1
                fold_cummulative_results["pairs_links"]["gold"] += 1


            predicted_expansion, module_used = predicted_expansions.pop(
                acronym, (None, None)
            )
            # try removing s or add s
            if (
                predicted_expansion == (None, None)
                and len(acronym) > 1
                and acronym[-1] == "s"
            ):
                predicted_expansion, module_used = predicted_expansions.pop(
                    acronym[:-1], (None, None)
                )
            elif predicted_expansion == (None, None) and acronym[-1] != "s":
                predicted_expansion, module_used = predicted_expansions.pop(
                    acronym + "s", (None, None)
                )

            if predicted_expansion:            
                options = []
                if isinstance(predicted_expansion, AcronymExpansion):
                    options = predicted_expansion.options
                    predicted_expansion = predicted_expansion.expansion
                
                
                fold_cummulative_results["acronyms_total"]["extracted"] += 1
                fold_cummulative_results["expansions_total"]["extracted"] += 1
                fold_cummulative_results["pairs_total"]["extracted"] += 1
                                    
                fold_cummulative_results["acronyms_total"]["correct"] += 1
                
                
                metric_suffix = self._acroexp_module_to_metric_suffix(module_used)
                fold_cummulative_results["acronyms_" + metric_suffix]["extracted"] += 1
                fold_cummulative_results["expansions_" + metric_suffix]["extracted"] += 1
                fold_cummulative_results["pairs_" + metric_suffix]["extracted"] += 1

                fold_cummulative_results["acronyms_" + metric_suffix]["correct"] += 1
                
                if module_used == ExpansionModuleEnum.out_expander and in_or_out_text == "in":
                    fold_cummulative_results["acronyms_out"]["gold"] += 1
                    fold_cummulative_results["expansions_out"]["gold"] += 1
                    fold_cummulative_results["pairs_out"]["gold"] += 1
            


                if AcronymExpansion.areExpansionsSimilar(
                    actual_expansion.strip().lower(),
                    predicted_expansion.strip().lower(),
                ):

                    fold_cummulative_results["expansions_" + metric_suffix][
                        "correct"
                    ] += 1
                    fold_cummulative_results["pairs_" + metric_suffix]["correct"] += 1

                    fold_cummulative_results["expansions_total"][
                        "correct"
                    ] += 1
                    fold_cummulative_results["pairs_total"]["correct"] += 1

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
                                        in_text,
                                        module_used.name,
                                    ],
                                )
                            )
                        )

                else:
                    if self._simExpansionExists(
                            actual_expansion, options
                        ) == False:
                            # Here we keep track how many error predictions were given 
                            #because the true expansion is not in our DB.
                            fold_cummulative_results["incorrect_missing_label"] += 1
                        

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
                                        in_text,
                                        module_used.name,
                                    ],
                                )
                            )
                        )
            elif in_or_out_text == "in": # Case where in-expander fails and also out-expander to identify and predict some acronym
                fold_cummulative_results["acronyms_out"]["gold"] += 1
                fold_cummulative_results["expansions_out"]["gold"] += 1
                fold_cummulative_results["pairs_out"]["gold"] += 1
 
                results_writer(
                    dict(
                        zip(
                            QUALITY_FIELDS_PER_ARTICLE,
                            [
                                acronym,
                                actual_expansion,
                                "",
                                False,
                                in_text,
                                None,
                            ],
                        )
                    )
                )    
             
            else:
                
                results_writer(
                    dict(
                        zip(
                            QUALITY_FIELDS_PER_ARTICLE,
                            [
                                acronym,
                                actual_expansion,
                                "",
                                False,
                                in_text,
                                None,
                            ],
                        )
                    )
                )         
                   
        for acronym, [predicted_expansion, module_used] in predicted_expansions.items():
            
            if isinstance(predicted_expansion, AcronymExpansion):
                options = predicted_expansion.options
                predicted_expansion = predicted_expansion.expansion
            
            metric_suffix = self._acroexp_module_to_metric_suffix(module_used)
            
            fold_cummulative_results["acronyms_" + metric_suffix]["extracted"] += 1
            fold_cummulative_results["pairs_" + metric_suffix]["extracted"] += 1
            
            fold_cummulative_results["acronyms_total"]["extracted"] += 1
            fold_cummulative_results["pairs_total"]["extracted"] += 1

            logger.debug(
                "Expansion matching failed in doc_id %s, (%s): %s, %s"
                % (doc_id, acronym, None, predicted_expansion)
            )

            if results_writer:

                results_writer(
                    dict(
                        zip(
                            QUALITY_FIELDS_PER_ARTICLE,
                            [
                                acronym,
                                "",
                                predicted_expansion,
                                False,
                                None,
                                module_used.name,
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
        links_followed=[]
        if isinstance(predicted_expansions, collections.Sequence) and len(predicted_expansions) > 1:
            links_followed = predicted_expansions[1]
            predicted_expansions = predicted_expansions[0]
            fold_cummulative_results["total_links_followed"] += len(links_followed)

        actual_expansions_in = {
            k.lower(): v[0] for k, v in actual_expansions.items() if v[1]
        }
        actual_expansions_out = {
            k.lower(): v[0] for k, v in actual_expansions.items() if not v[1]
        }

        predicted_expansions_in = {
            k.lower(): v[0]
            for k, v in predicted_expansions.items()
            if v[1] == ExpansionModuleEnum.in_expander
        }

        
        predicted_expansions_out = {
            k.lower(): v[0]
            for k, v in predicted_expansions.items()
            if not v[1] == ExpansionModuleEnum.in_expander
        }
        

        if len(actual_expansions_in) != 0 and len(predicted_expansions_in) != 0:
            fold_cummulative_results["annotations_data_acro_in"].append(
                (("u1"), int(doc_id), frozenset(actual_expansions_in.keys()))
            )
            fold_cummulative_results["annotations_data_acro_in"].append(
                (("u2"), int(doc_id), frozenset(predicted_expansions_in.keys()))
            )
            fold_cummulative_results["annotations_data_exp_in"].append(
                (("u1"), int(doc_id), frozenset(actual_expansions_in.values()))
            )
            fold_cummulative_results["annotations_data_exp_in"].append(
                (("u2"), int(doc_id), frozenset(predicted_expansions_in.values()))
            )
            fold_cummulative_results["annotations_data_pairs_in"].append(
                (("u1"), int(doc_id), frozenset(actual_expansions_in.items()))
            )
            fold_cummulative_results["annotations_data_pairs_in"].append(
                (("u2"), int(doc_id), frozenset(predicted_expansions_in.items()))
            )

        if len(actual_expansions_out) != 0 and len(predicted_expansions_out) != 0:
            fold_cummulative_results["annotations_data_acro_out"].append(
                (("u1"), int(doc_id), frozenset(actual_expansions_out.keys()))
            )
            fold_cummulative_results["annotations_data_acro_out"].append(
                (("u2"), int(doc_id), frozenset(predicted_expansions_out.keys()))
            )
            fold_cummulative_results["annotations_data_exp_out"].append(
                (("u1"), int(doc_id), frozenset(actual_expansions_out.values()))
            )
            fold_cummulative_results["annotations_data_exp_out"].append(
                (("u2"), int(doc_id), frozenset(predicted_expansions_out.values()))
            )
            fold_cummulative_results["annotations_data_pairs_out"].append(
                (("u1"), int(doc_id), frozenset(actual_expansions_out.items()))
            )
            fold_cummulative_results["annotations_data_pairs_out"].append(
                (("u2"), int(doc_id), frozenset(predicted_expansions_out.items()))
            )

        if (len(actual_expansions_in) != 0 or len(actual_expansions_out) != 0) and (
            len(predicted_expansions_in) != 0 or len(predicted_expansions_out) != 0
        ):
            fold_cummulative_results["annotations_data_acro_total"].append(
                (
                    ("u1"),
                    int(doc_id),
                    frozenset(
                        list(actual_expansions_in.keys())
                        + list(actual_expansions_out.keys())
                    ),
                )
            )
            fold_cummulative_results["annotations_data_acro_total"].append(
                (
                    ("u2"),
                    int(doc_id),
                    frozenset(
                        list(predicted_expansions_in.keys())
                        + list(predicted_expansions_out.keys())
                    ),
                )
            )
            fold_cummulative_results["annotations_data_exp_total"].append(
                (
                    ("u1"),
                    int(doc_id),
                    frozenset(
                        list(actual_expansions_in.values())
                        + list(actual_expansions_out.values())
                    ),
                )
            )
            fold_cummulative_results["annotations_data_exp_total"].append(
                (
                    ("u2"),
                    int(doc_id),
                    frozenset(
                        list(predicted_expansions_in.values())
                        + list(predicted_expansions_out.values())
                    ),
                )
            )
            fold_cummulative_results["annotations_data_pairs_total"].append(
                (
                    ("u1"),
                    int(doc_id),
                    frozenset(
                        list(actual_expansions_in.items())
                        + list(actual_expansions_out.items())
                    ),
                )
            )
            fold_cummulative_results["annotations_data_pairs_total"].append(
                (
                    ("u2"),
                    int(doc_id),
                    frozenset(
                        list(predicted_expansions_in.items())
                        + list(predicted_expansions_out.items())
                    ),
                )
            )

        self._calculate_quality_results(
        fold_cummulative_results,
        results_writer,
        doc_id,
        actual_expansions,
        predicted_expansions,
        links_followed
        )
        
        
    def _compute_fold_results_precision_recall_f1(self, fold_results, element, exp_type):
            comb_name = element +"_" + exp_type
            try:
                fold_results["precision_"+ comb_name] = (
                    fold_results[comb_name]["correct"]
                    / fold_results[comb_name]["extracted"]
                )
            except ZeroDivisionError:
                fold_results["precision_"+ comb_name] = 0
    
            try:
                fold_results["recall_"+comb_name] = (
                    fold_results[comb_name]["correct"]
                    / fold_results[comb_name]["gold"]
                )
            except ZeroDivisionError:
                fold_results["recall_"+comb_name] = 0
    
            try:
                fold_results["f1_score_"+comb_name] = 2 * (
                    (
                        fold_results["precision_"+comb_name]
                        * fold_results["recall_"+comb_name]
                    )
                    / (
                        fold_results["precision_"+comb_name]
                        + fold_results["recall_"+comb_name]
                    )
                )
            except ZeroDivisionError:
                fold_results["f1_score_"+comb_name] = 0
        
        
    def _compute_individual_fold_quality_results(self, fold, fold_results):
        element_type = ["acronyms", "expansions", "pairs"]
        expansion_type = ["in","links","out","total"]
        
        all_combinations = itertools.product(element_type, expansion_type)
            
        for element, exp_type in all_combinations:
            self._compute_fold_results_precision_recall_f1(fold_results, element, exp_type)

    
    def _compute_fold_quality_for_exp_in_db(self, fold, fold_results):
            element_type = ["acronyms", "expansions", "pairs"]
            expansion_type = ["out", "total"]
            aditional_suffix = "_in_db"
            all_combinations = itertools.product(element_type, expansion_type)
            for element, exp_type in all_combinations:
                comb_name = element + "_" + exp_type
                name_in_db = comb_name + aditional_suffix
                
                fold_results[name_in_db] = {}
                fold_results[name_in_db]["gold"] = fold_results[comb_name]["gold"] - fold_results["incorrect_missing_label"]
                fold_results[name_in_db]["extracted"] = fold_results[comb_name]["extracted"] - fold_results["incorrect_missing_label"]
                fold_results[name_in_db]["correct"] = fold_results[comb_name]["correct"]
                self._compute_fold_results_precision_recall_f1(fold_results, element, exp_type+aditional_suffix)
    
    # TODO refactoring
    def _compute_fold_quality_results(self, fold, fold_results):
        """Calculates all final metrics for the current fold based on the cumulative results.

        Args:
            fold (str):
             the fold for which metrics are going to be calculated
            fold_results (dict):
             a dictionary where all the metrics are stored.
        """
        self._compute_individual_fold_quality_results(fold, fold_results)

        self._compute_fold_quality_for_exp_in_db(fold, fold_results)


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

        task_acro_out = agreement.AnnotationTask(
            data=fold_results["annotations_data_acro_out"],
            distance=jaccard_distance,
        )

        task_exp_out = agreement.AnnotationTask(
            data=fold_results["annotations_data_exp_out"],
            distance=jaccard_distance,
        )

        task_pairs_out = agreement.AnnotationTask(
            data=fold_results["annotations_data_pairs_out"],
            distance=jaccard_distance,
        )

        if len(fold_results["annotations_data_acro_out"]) != 0:
            fold_results["alpha_jaccard_acro_out"] = task_acro_out.alpha()
            fold_results["kappa_jaccard_acro_out"] = task_acro_out.kappa()
        else:
            fold_results["alpha_jaccard_acro_out"] = 0
            fold_results["kappa_jaccard_acro_out"] = 0

        if len(fold_results["annotations_data_exp_out"]) != 0:
            fold_results["alpha_jaccard_exp_out"] = task_exp_out.alpha()
            fold_results["kappa_jaccard_exp_out"] = task_exp_out.kappa()
        else:
            fold_results["alpha_jaccard_exp_out"] = 0
            fold_results["kappa_jaccard_exp_out"] = 0

        if len(fold_results["annotations_data_pairs_out"]) != 0:
            fold_results["alpha_jaccard_pairs_out"] = task_pairs_out.alpha()
            fold_results["kappa_jaccard_pairs_out"] = task_pairs_out.kappa()
        else:
            fold_results["alpha_jaccard_pairs_out"] = 0
            fold_results["kappa_jaccard_pairs_out"] = 0

        task_acro_out = agreement.AnnotationTask(
            data=fold_results["annotations_data_acro_out"],
            distance=masi_distance,
        )

        task_exp_out = agreement.AnnotationTask(
            data=fold_results["annotations_data_exp_out"],
            distance=masi_distance,
        )

        task_pairs_out = agreement.AnnotationTask(
            data=fold_results["annotations_data_pairs_out"],
            distance=masi_distance,
        )

        if len(fold_results["annotations_data_acro_out"]) != 0:
            fold_results["alpha_masi_acro_out"] = task_acro_out.alpha()
            fold_results["kappa_masi_acro_out"] = task_acro_out.kappa()
        else:
            fold_results["alpha_masi_acro_out"] = 0
            fold_results["kappa_masi_acro_out"] = 0

        if len(fold_results["annotations_data_exp_out"]) != 0:
            fold_results["alpha_masi_exp_out"] = task_exp_out.alpha()
            fold_results["kappa_masi_exp_out"] = task_exp_out.kappa()
        else:
            fold_results["alpha_masi_exp_out"] = 0
            fold_results["kappa_masi_exp_out"] = 0

        if len(fold_results["annotations_data_pairs_out"]) != 0:
            fold_results["alpha_masi_pairs_out"] = task_pairs_out.alpha()
            fold_results["kappa_masi_pairs_out"] = task_pairs_out.kappa()
        else:
            fold_results["alpha_masi_pairs_out"] = 0
            fold_results["kappa_masi_pairs_out"] = 0

        task_acro_total = agreement.AnnotationTask(
            data=fold_results["annotations_data_acro_total"],
            distance=jaccard_distance,
        )

        task_exp_total = agreement.AnnotationTask(
            data=fold_results["annotations_data_exp_total"],
            distance=jaccard_distance,
        )

        task_pairs_total = agreement.AnnotationTask(
            data=fold_results["annotations_data_pairs_total"],
            distance=jaccard_distance,
        )

        if len(fold_results["annotations_data_acro_total"]) != 0:
            fold_results["alpha_jaccard_acro_total"] = task_acro_total.alpha()
            fold_results["kappa_jaccard_acro_total"] = task_acro_total.kappa()
        else:
            fold_results["alpha_jaccard_acro_total"] = 0
            fold_results["kappa_jaccard_acro_total"] = 0

        if len(fold_results["annotations_data_exp_total"]) != 0:
            fold_results["alpha_jaccard_exp_total"] = task_exp_total.alpha()
            fold_results["kappa_jaccard_exp_total"] = task_exp_total.kappa()
        else:
            fold_results["alpha_jaccard_exp_total"] = 0
            fold_results["kappa_jaccard_exp_total"] = 0

        if len(fold_results["annotations_data_pairs_total"]) != 0:
            fold_results["alpha_jaccard_pairs_total"] = task_pairs_total.alpha()
            fold_results["kappa_jaccard_pairs_total"] = task_pairs_total.kappa()
        else:
            fold_results["alpha_jaccard_pairs_total"] = 0
            fold_results["kappa_jaccard_pairs_total"] = 0

        task_acro_total = agreement.AnnotationTask(
            data=fold_results["annotations_data_acro_total"],
            distance=masi_distance,
        )

        task_exp_total = agreement.AnnotationTask(
            data=fold_results["annotations_data_exp_total"],
            distance=masi_distance,
        )

        task_pairs_total = agreement.AnnotationTask(
            data=fold_results["annotations_data_pairs_total"],
            distance=masi_distance,
        )

        if len(fold_results["annotations_data_acro_total"]) != 0:
            fold_results["alpha_masi_acro_total"] = task_acro_total.alpha()
            fold_results["kappa_masi_acro_total"] = task_acro_total.kappa()
        else:
            fold_results["alpha_masi_acro_total"] = 0
            fold_results["kappa_masi_acro_total"] = 0

        if len(fold_results["annotations_data_exp_total"]) != 0:
            fold_results["alpha_masi_exp_total"] = task_exp_total.alpha()
            fold_results["kappa_masi_exp_total"] = task_exp_total.kappa()
        else:
            fold_results["alpha_masi_exp_total"] = 0
            fold_results["kappa_masi_exp_total"] = 0

        if len(fold_results["annotations_data_pairs_total"]) != 0:
            fold_results["alpha_masi_pairs_total"] = task_pairs_total.alpha()
            fold_results["kappa_masi_pairs_total"] = task_pairs_total.kappa()
        else:
            fold_results["alpha_masi_pairs_total"] = 0
            fold_results["kappa_masi_pairs_total"] = 0

    def _get_metric_fold_mean(self, metric_name):
        non_none_fold_metric_values = [
            item[metric_name]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        return sum(non_none_fold_metric_values) / len(self.cumulativeResults)

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

        sum_pairs_in_gold = sum(
            [
                item["pairs_in"]["gold"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        sum_pairs_in_correct = sum(
            [
                item["pairs_in"]["correct"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        sum_pairs_in_extracted = sum(
            [
                item["pairs_in"]["extracted"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        try:
            micro_precision_pairs_in = sum_pairs_in_correct / sum_pairs_in_extracted

        except ZeroDivisionError:
            micro_precision_pairs_in = 0

        try:
            micro_recall_pairs_in = sum_pairs_in_correct / sum_pairs_in_gold
        except ZeroDivisionError:
            micro_recall_pairs_in = 0

        try:
            micro_f1_pairs_in = 2 * (
                (micro_precision_pairs_in * micro_recall_pairs_in)
                / (micro_precision_pairs_in + micro_recall_pairs_in)
            )
        except ZeroDivisionError:
            micro_f1_pairs_in = 0

        logger.critical(
            "Pairs in text -> Precision: %f, Recall: %f, F1-Measure: %f, Micro Precision: %f, Micro Recall: %f, Micro F1-Measure: %f",
            mean_precision_pairs_in,
            mean_recall_pairs_in,
            mean_f1_score_pairs_in,
            micro_precision_pairs_in,
            micro_recall_pairs_in,
            micro_f1_pairs_in,
        )

        precision_acronyms_out = [
            item["precision_acronyms_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_acronyms_out = mean(precision_acronyms_out)

        recall_acronyms_out = [
            item["recall_acronyms_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_acronyms_out = mean(recall_acronyms_out)

        f1_score_acronyms_out = [
            item["f1_score_acronyms_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_acronyms_out = mean(f1_score_acronyms_out)

        logger.critical(
            "Acronyms with no expansion in text -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_precision_acronyms_out,
            mean_recall_acronyms_out,
            mean_f1_score_acronyms_out,
        )

        precision_expansions_out = [
            item["precision_expansions_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_expansions_out = mean(precision_expansions_out)

        recall_expansions_out = [
            item["recall_expansions_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_expansions_out = mean(recall_expansions_out)

        f1_score_expansions_out = [
            item["f1_score_expansions_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_expansions_out = mean(f1_score_expansions_out)

        logger.critical(
            "Expansions not in text -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_precision_expansions_out,
            mean_recall_expansions_out,
            mean_f1_score_expansions_out,
        )

        precision_pairs_out = [
            item["precision_pairs_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_pairs_out = mean(precision_pairs_out)

        recall_pairs_out = [
            item["recall_pairs_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_pairs_out = mean(recall_pairs_out)

        f1_score_pairs_out = [
            item["f1_score_pairs_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_pairs_out = mean(f1_score_pairs_out)

        sum_pairs_out_gold = sum(
            [
                item["pairs_out"]["gold"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        sum_pairs_out_correct = sum(
            [
                item["pairs_out"]["correct"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        sum_pairs_out_extracted = sum(
            [
                item["pairs_out"]["extracted"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        try:
            micro_precision_pairs_out = sum_pairs_out_correct / sum_pairs_out_extracted
        except ZeroDivisionError:
            micro_precision_pairs_out = 0

        try:
            micro_recall_pairs_out = sum_pairs_out_correct / sum_pairs_out_gold
        except ZeroDivisionError:
            micro_recall_pairs_out = 0

        try:
            micro_f1_pairs_out = 2 * (
                (micro_precision_pairs_out * micro_recall_pairs_out)
                / (micro_precision_pairs_out + micro_recall_pairs_out)
            )
        except ZeroDivisionError:
            micro_f1_pairs_out = 0

        logger.critical(
            "Pairs not in text -> Precision: %f, Recall: %f, F1-Measure: %f, Micro Precision: %f, Micro Recall: %f, Micro F1-Measure: %f",
            mean_precision_pairs_out,
            mean_recall_pairs_out,
            mean_f1_score_pairs_out,
            micro_precision_pairs_out,
            micro_recall_pairs_out,
            micro_f1_pairs_out,
        )

        precision_acronyms_total = [
            item["precision_acronyms_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_acronyms_total = mean(precision_acronyms_total)

        recall_acronyms_total = [
            item["recall_acronyms_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_acronyms_total = mean(recall_acronyms_total)

        f1_score_acronyms_total = [
            item["f1_score_acronyms_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_acronyms_total = mean(f1_score_acronyms_total)

        logger.critical(
            "All acronyms -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_precision_acronyms_total,
            mean_recall_acronyms_total,
            mean_f1_score_acronyms_total,
        )

        precision_expansions_total = [
            item["precision_expansions_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_expansions_total = mean(precision_expansions_total)

        recall_expansions_total = [
            item["recall_expansions_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_expansions_total = mean(recall_expansions_total)

        f1_score_expansions_total = [
            item["f1_score_expansions_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_expansions_total = mean(f1_score_expansions_total)

        logger.critical(
            "All expansions -> Precision: %f, Recall: %f, F1-Measure: %f",
            mean_precision_expansions_total,
            mean_recall_expansions_total,
            mean_f1_score_expansions_total,
        )

        precision_pairs_total = [
            item["precision_pairs_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_precision_pairs_total = mean(precision_pairs_total)

        recall_pairs_total = [
            item["recall_pairs_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_recall_pairs_total = mean(recall_pairs_total)

        f1_score_pairs_total = [
            item["f1_score_pairs_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_f1_score_pairs_total = mean(f1_score_pairs_total)

        sum_pairs_total_gold = sum(
            [
                item["pairs_in"]["gold"] + item["pairs_out"]["gold"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        sum_pairs_total_correct = sum(
            [
                item["pairs_in"]["correct"] + item["pairs_out"]["correct"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        sum_pairs_total_extracted = sum(
            [
                item["pairs_in"]["extracted"] + item["pairs_out"]["extracted"]
                for item in self.cumulativeResults.values()
                if item != None
            ]
        )

        try:
            micro_precision_pairs_total = (
                sum_pairs_total_correct / sum_pairs_total_extracted
            )
        except ZeroDivisionError:
            micro_precision_pairs_total = 0

        try:
            micro_recall_pairs_total = sum_pairs_total_correct / sum_pairs_total_gold
        except ZeroDivisionError:
            micro_recall_pairs_total = 0

        try:
            micro_f1_pairs_total = 2 * (
                (micro_precision_pairs_total * micro_recall_pairs_total)
                / (micro_precision_pairs_total + micro_recall_pairs_total)
            )
        except ZeroDivisionError:
            micro_f1_pairs_total = 0

        logger.critical(
            "All pairs -> Precision: %f, Recall: %f, F1-Measure: %f, Micro Precision: %f, Micro Recall: %f, Micro F1-Measure: %f",
            mean_precision_pairs_total,
            mean_recall_pairs_total,
            mean_f1_score_pairs_total,
            micro_precision_pairs_total,
            micro_recall_pairs_total,
            micro_f1_pairs_total,
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

        alpha_jaccard_acro_out = [
            item["alpha_jaccard_acro_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_jaccard_exp_out = [
            item["alpha_jaccard_exp_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_jaccard_pairs_out = [
            item["alpha_jaccard_pairs_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_acro_out = [
            item["kappa_jaccard_acro_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_exp_out = [
            item["kappa_jaccard_exp_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_pairs_out = [
            item["kappa_jaccard_pairs_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        alpha_masi_acro_out = [
            item["alpha_masi_acro_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_masi_exp_out = [
            item["alpha_masi_exp_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_masi_pairs_out = [
            item["alpha_masi_pairs_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_acro_out = [
            item["kappa_masi_acro_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_exp_out = [
            item["kappa_masi_exp_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_pairs_out = [
            item["kappa_masi_pairs_out"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        alpha_jaccard_acro_total = [
            item["alpha_jaccard_acro_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_jaccard_exp_total = [
            item["alpha_jaccard_exp_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_jaccard_pairs_total = [
            item["alpha_jaccard_pairs_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_acro_total = [
            item["kappa_jaccard_acro_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_exp_total = [
            item["kappa_jaccard_exp_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_jaccard_pairs_total = [
            item["kappa_jaccard_pairs_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        alpha_masi_acro_total = [
            item["alpha_masi_acro_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_masi_exp_total = [
            item["alpha_masi_exp_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        alpha_masi_pairs_total = [
            item["alpha_masi_pairs_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_acro_total = [
            item["kappa_masi_acro_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_exp_total = [
            item["kappa_masi_exp_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        kappa_masi_pairs_total = [
            item["kappa_masi_pairs_total"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        
        total_links_followed = [
            fold_cummulative_results["total_links_followed"]
            for fold_cummulative_results in self.cumulativeResults.values()
            if fold_cummulative_results != None]
        total_links_followed = sum(total_links_followed)
        logger.critical("Total Links Followed: %d",  total_links_followed)
        
        
        
        
        measure_values = [
            mean_precision_acronyms_total,
            mean_recall_acronyms_total,
            mean_f1_score_acronyms_total,
            mean(alpha_jaccard_acro_total),
            mean(alpha_masi_acro_total),
            mean(kappa_jaccard_acro_total),
            mean(kappa_masi_acro_total),
            mean_precision_expansions_total,
            mean_recall_expansions_total,
            mean_f1_score_expansions_total,
            mean(alpha_jaccard_exp_total),
            mean(alpha_masi_exp_total),
            mean(kappa_jaccard_exp_total),
            mean(kappa_masi_exp_total),
            mean_precision_pairs_total,
            mean_recall_pairs_total,
            mean_f1_score_pairs_total,
            micro_precision_pairs_total,
            micro_recall_pairs_total,
            micro_f1_pairs_total,
            mean(alpha_jaccard_pairs_total),
            mean(alpha_masi_pairs_total),
            mean(kappa_jaccard_pairs_total),
            mean(kappa_masi_pairs_total),
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
            micro_precision_pairs_in,
            micro_recall_pairs_in,
            micro_f1_pairs_in,
            mean(alpha_jaccard_pairs_in),
            mean(alpha_masi_pairs_in),
            mean(kappa_jaccard_pairs_in),
            mean(kappa_masi_pairs_in),
            mean_precision_acronyms_out,
            mean_recall_acronyms_out,
            mean_f1_score_acronyms_out,
            mean(alpha_jaccard_acro_out),
            mean(alpha_masi_acro_out),
            mean(kappa_jaccard_acro_out),
            mean(kappa_masi_acro_out),
            mean_precision_expansions_out,
            mean_recall_expansions_out,
            mean_f1_score_expansions_out,
            mean(alpha_jaccard_exp_out),
            mean(alpha_masi_exp_out),
            mean(kappa_jaccard_exp_out),
            mean(kappa_masi_exp_out),
            mean_precision_pairs_out,
            mean_recall_pairs_out,
            mean_f1_score_pairs_out,
            micro_precision_pairs_out,
            micro_recall_pairs_out,
            micro_f1_pairs_out,
            mean(alpha_jaccard_pairs_out),
            mean(alpha_masi_pairs_out),
            mean(kappa_jaccard_pairs_out),
            mean(kappa_masi_pairs_out),
            # TODO prints of metrics below
            self._get_metric_fold_mean("precision_acronyms_links"),
            self._get_metric_fold_mean("recall_acronyms_links"),
            self._get_metric_fold_mean("f1_score_acronyms_links"),
            
            self._get_metric_fold_mean("precision_expansions_links"),
            self._get_metric_fold_mean("recall_expansions_links"),
            self._get_metric_fold_mean("f1_score_expansions_links"),
            
            self._get_metric_fold_mean("precision_pairs_links"),
            self._get_metric_fold_mean("recall_pairs_links"),
            self._get_metric_fold_mean("f1_score_pairs_links"),
            
            self._get_metric_fold_mean("precision_acronyms_out_in_db"),
            self._get_metric_fold_mean("recall_acronyms_out_in_db"),
            self._get_metric_fold_mean("f1_score_acronyms_out_in_db"),
            
            self._get_metric_fold_mean("precision_expansions_out_in_db"),
            self._get_metric_fold_mean("recall_expansions_out_in_db"),
            self._get_metric_fold_mean("f1_score_expansions_out_in_db"),
            
            self._get_metric_fold_mean("precision_pairs_out_in_db"),
            self._get_metric_fold_mean("recall_pairs_out_in_db"),
            self._get_metric_fold_mean("f1_score_pairs_out_in_db"),
            
            self._get_metric_fold_mean("precision_acronyms_total_in_db"),
            self._get_metric_fold_mean("recall_acronyms_total_in_db"),
            self._get_metric_fold_mean("f1_score_acronyms_total_in_db"),
            
            self._get_metric_fold_mean("precision_expansions_total_in_db"),
            self._get_metric_fold_mean("recall_expansions_total_in_db"),
            self._get_metric_fold_mean("f1_score_expansions_total_in_db"),
            
            self._get_metric_fold_mean("precision_pairs_total_in_db"),
            self._get_metric_fold_mean("recall_pairs_total_in_db"),
            self._get_metric_fold_mean("f1_score_pairs_total_in_db"),
            
            total_links_followed,
        ]

        return dict(zip(QUALITY_FIELDS_LIST + ["Total Links Followed"], measure_values))
