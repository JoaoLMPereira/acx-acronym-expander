"""
Created on Feb 20, 2020

@author: jpereira
"""
from collections import defaultdict
import csv
import json
import os
from numpy import mean, std

from Logger import logging
from benchmarkers.results_reporter_base import ResultsReporterBase
from helper import AcronymExpansion, flatten_to_str_list
from string_constants import REPORT_OUT_EXPANSION_NAME, FOLDER_LOGS


logger = logging.getLogger(__name__)

GENERAL_REPORT_FIELDS = {
    "out_expander": "VARCHAR(255)",
    "out_expander_args": "VARCHAR(2000)",
}

SQL_FRACTION_TYPE = "DECIMAL(6, 5)"

QUALITY_FIELDS_LIST = [
    "Accuracy Micro Fold",
    "Accuracy Fold Mean",
    "Accuracy Fold Std Dev",
    "Exp in DB Accuracy Micro Fold",
    "Exp in DB Accuracy Fold Mean",
    "Exp in DB Accuracy Fold Std Dev",
    "Ambiguous Acronym Accuracy Micro Fold",
    "Ambiguous Acronym Accuracy Fold Mean",
    "Ambiguous Acronym Accuracy Fold Std Dev",
    "Ambiguous Acronym Exp in DB Accuracy Micro Fold",
    "Ambiguous Acronym Exp in DB Accuracy Fold Mean",
    "Ambiguous Acronym Exp in DB Accuracy Fold Std Dev",
    "Macro Precision",
    "Macro Recall",
    "Macro F1-Measure",
    "Macro Ambiguous Acronym Exp in DB Precision",
    "Macro Ambiguous Acronym Exp in DB Recall",
    "Macro Ambiguous Acronym Exp in DB F1-Measure",
]

QUALITY_FIELDS = {field_name: SQL_FRACTION_TYPE for field_name in QUALITY_FIELDS_LIST}

QUALITY_FIELDS_PER_ARTICLE = {
    "acronym": "VARCHAR(10)",
    "actual_expansion": "VARCHAR(255)",
    "predicted_expansion": "VARCHAR(255)",
    "confidence": "DECIMAL(12,10)",
    "success": "BOOLEAN",
}

REPORT_CONFIDENCES_FIELDS = ["fold", "doc_id", "acronym",
                              "confidences_json_dict"]


class ResultsReporter(ResultsReporterBase):
    """
    classdocs
    """

    def __init__(
        self,
        experiment_name: str,
        experiment_parameters,
        report_name: str = REPORT_OUT_EXPANSION_NAME,
        save_results_per_article: bool = False,
        db_config=None,
    ):
        """
        Constructor
        """

        cummulative_results_init = {
            "correct_expansions": 0,
            "incorrect_expansions": 0,
            "incorrect_missing_expansion": 0,
            "correct_disambiguations": 0,
            "incorrect_disambiguations": 0,
            "incorrect_missing_label": 0,
            "correct_per_expansion": defaultdict(int),
            "total_per_expansion": defaultdict(int),
            "pred_per_expansion": defaultdict(int),
            "ambiguous_acronym_expansions_in_train_db": set(),
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
        
        self.confidences_table_writter = None
        
    def _get_confidences_table_writter(self):
        if self.confidences_table_writter:
            return self.confidences_table_writter
        
        exp_writer = None
        if "_CV" in self.experiment_name:
            cv_expansions_file_name = FOLDER_LOGS + "expansions_" + self.experiment_name + ".csv"
            if not os.path.exists(cv_expansions_file_name):
                exp_file = open(cv_expansions_file_name, "w")
                exp_writer = csv.DictWriter(exp_file,
                                            fieldnames=["fold", "doc_id", "acronym", "expansion"], 
                                            extrasaction='ignore')
                exp_writer.writeheader()
        
        string_exp_param = "_".join(flatten_to_str_list(self.experiment_parameters))
        
        f = open(FOLDER_LOGS + "confidences_" + self.experiment_name + "_" + string_exp_param + ".csv", "w")
        writer = csv.DictWriter(
            f,
            fieldnames= REPORT_CONFIDENCES_FIELDS
        )
        if f.tell() == 0:
            writer.writeheader()
        
        def writer_row_confidences(arg_fields, actual_expansion):
            writer.writerow(arg_fields)
            if exp_writer:
                exp_writer.writerow({**arg_fields, "expansion":actual_expansion})
            
        self.confidences_table_writter = writer_row_confidences
        return self.confidences_table_writter
    
    def _write_confidences(self, fold, doc_id, acronym, str_json_dict, actual_expansion):
        row_writter = self._get_confidences_table_writter()
        row_dict= {"fold": fold, "doc_id":doc_id,"acronym": acronym, "confidences_json_dict": json.dumps(str_json_dict)}
        row_writter(row_dict, actual_expansion)

    def _simExpansionExists(self, testExp, expList):
        for candidateExp in expList:
            if AcronymExpansion.areExpansionsSimilar(testExp, candidateExp):
                return True

        return False

    def _process_quality_results(
        self,
        fold_cummulative_results,
        results_writer,
        fold,
        doc_id,
        actual_expansions,
        predicted_expansions,
    ):

        for acronym in actual_expansions:
            actual_expansion = actual_expansions[acronym]
            fold_cummulative_results["total_per_expansion"][actual_expansion] += 1


            if acronym in predicted_expansions:
                options = predicted_expansions[acronym].options
                option_confidences = None
                if isinstance(options, dict):
                    #save to file
                    option_confidences = options
                    options = [opt[0] for opt in options.items()]
            # TODO replace app duplicate comparison by equality, after making sure every dataset was expansion linkage
            if (
                acronym in predicted_expansions
                and AcronymExpansion.areExpansionsSimilar(
                    actual_expansion, predicted_expansions[acronym].expansion
                )
            ):
                
                logger.debug(
                    "Expansion matching succeeded (%s): %s, %s, confidence: %f, options %r",
                    acronym,
                    actual_expansion,
                    predicted_expansions[acronym].expansion,
                    predicted_expansions[acronym].confidence,
                    ";".join(options),
                )
                fold_cummulative_results["correct_expansions"] += 1
                fold_cummulative_results["correct_per_expansion"][actual_expansion] += 1
                fold_cummulative_results["pred_per_expansion"][actual_expansion] += 1

                if len(options) > 1:
                    fold_cummulative_results["correct_disambiguations"] += 1
                    fold_cummulative_results[
                        "ambiguous_acronym_expansions_in_train_db"
                    ].add(actual_expansion)
                    if results_writer:
                        results_writer(
                            dict(
                                zip(
                                    QUALITY_FIELDS_PER_ARTICLE,
                                    [
                                        acronym,
                                        actual_expansion,
                                        predicted_expansions[acronym].expansion,
                                        predicted_expansions[
                                            acronym
                                        ].confidence,
                                        True,
                                    ],
                                )
                            )
                        )
                    if option_confidences:
                        self._write_confidences(fold, doc_id, acronym, option_confidences, actual_expansion)

            elif acronym in predicted_expansions:
                predicted_expansion = predicted_expansions[acronym].expansion
                confidence = predicted_expansions[acronym].confidence

                fold_cummulative_results["pred_per_expansion"][predicted_expansion] += 1
                fold_cummulative_results["incorrect_expansions"] += 1

                if len(options) > 1:
                    fold_cummulative_results["incorrect_disambiguations"] += 1

                    if (
                        self._simExpansionExists(
                            actual_expansion, predicted_expansions[acronym].options
                        )
                        == False
                    ):
                        fold_cummulative_results["incorrect_missing_label"] += 1
                        fold_cummulative_results["incorrect_missing_expansion"] += 1
                        
                    else:

                        fold_cummulative_results[
                            "ambiguous_acronym_expansions_in_train_db"
                        ].add(predicted_expansions[acronym].expansion)
                        fold_cummulative_results[
                            "ambiguous_acronym_expansions_in_train_db"
                        ].add(actual_expansion)

                        if results_writer:
                            results_writer(
                                dict(
                                    zip(
                                        QUALITY_FIELDS_PER_ARTICLE,
                                        [
                                            acronym,
                                            actual_expansion,
                                            predicted_expansions[acronym].expansion,
                                            confidence,
                                            False,
                                        ],
                                    )
                                )
                            )
                            
                        if option_confidences:
                            self._write_confidences(fold, doc_id, acronym, option_confidences, actual_expansion)
                logger.debug(
                    "Expansion matching failed (%s): %s, %s, confidence: %f, options: %r",
                    acronym,
                    actual_expansion,
                    predicted_expansion,
                    confidence,
                    ";".join(options),
                )

            else:
                logger.debug(
                    "Expansion matching failed (%s): %s, no option available",
                    acronym,
                    actual_expansion,
                )

                # predicted_expansion = predicted_expansions[acronym].expansion
                # fold_cummulative_results["pred_per_expansion"][
                #    predicted_expansion
                # ] += 1
                fold_cummulative_results["incorrect_expansions"] += 1
                fold_cummulative_results["incorrect_missing_expansion"] += 1

            # else:
            #    logger.error("Expansion not predicted for %s", acronym)

    def _compute_fold_quality_results(self, fold, fold_results):
        totalExpansions = (
            fold_results["correct_expansions"] + fold_results["incorrect_expansions"]
        )
        totalDisambiguations = (
            fold_results["correct_disambiguations"]
            + fold_results["incorrect_disambiguations"]
        )

        incorrect_missing_expansion = fold_results["incorrect_missing_expansion"]

        fold_results["total_expansions"] = totalExpansions
        if totalExpansions > 0:
            fold_results["average_correct_expansions"] = (
                fold_results["correct_expansions"] / totalExpansions
            )
            fold_results["average_incorrect_expansions"] = (
                fold_results["incorrect_expansions"] / totalExpansions
            )

            fold_results["average_possible_correct_expansions"] = fold_results[
                "correct_expansions"
            ] / (totalExpansions - incorrect_missing_expansion)

        fold_results["total_disambiguations"] = totalDisambiguations

        # Compute macro avg
        precs = defaultdict(int)
        recalls = defaultdict(int)
        f1_measures = defaultdict(int)

        total_per_expansion = fold_results["total_per_expansion"]
        correct_per_expansion = fold_results["correct_per_expansion"]
        pred_per_expansion = fold_results["pred_per_expansion"]

        for exp in total_per_expansion.keys():
            precs[exp] = (
                correct_per_expansion[exp] / pred_per_expansion[exp]
                if exp in pred_per_expansion
                else 1
            )
            recalls[exp] = correct_per_expansion[exp] / total_per_expansion[exp]
            f1_measures[exp] = (
                2 * precs[exp] * recalls[exp] / (precs[exp] + recalls[exp])
                if precs[exp] + recalls[exp] != 0
                else 0
            )

        fold_results["macro_prec"] = sum(precs.values()) / len(precs) if precs else 0
        fold_results["macro_recall"] = sum(recalls.values()) / len(recalls) if recalls else 0
        fold_results["macro_f1_measure"] = sum(f1_measures.values()) / len(f1_measures) if f1_measures else 0

        amb_macro_prec_mean = mean(
            [
                value
                for key, value in precs.items()
                if key in fold_results["ambiguous_acronym_expansions_in_train_db"]
            ]
        )
        fold_results["macro_prec_ambiguous_acro_exp_in_db"] = (
            amb_macro_prec_mean if amb_macro_prec_mean else 0
        )
        amb_macro_recall_mean = mean(
            [
                value
                for key, value in recalls.items()
                if key in fold_results["ambiguous_acronym_expansions_in_train_db"]
            ]
        )

        fold_results["macro_recall_ambiguous_acro_exp_in_db"] = (
            amb_macro_recall_mean if amb_macro_recall_mean else 0
        )
        amb_macro_f1_mean = mean(
            [
                value
                for key, value in f1_measures.items()
                if key in fold_results["ambiguous_acronym_expansions_in_train_db"]
            ]
        )
        fold_results["macro_f1_measure_ambiguous_acro_exp_in_db"] = (
            amb_macro_f1_mean if amb_macro_f1_mean else 0
        )

    def _plot_quality_stats(self):

        validSuccesses = [
            item["average_correct_expansions"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        mean_val = mean(validSuccesses)
        std_dev = std(validSuccesses)
        logger.critical("Mean %f, Standard Deviation: %f", mean_val, std_dev)

        validCorrectExpansions = [
            item["correct_expansions"]
            for item in self.cumulativeResults.values()
            if item != None
        ]
        sumCorrectExpansions = sum(validCorrectExpansions)

        validExpansions = [
            item["total_expansions"]
            for item in self.cumulativeResults.values()
            if item is not None
        ]
        sumExpansions = sum(validExpansions)

        if sumExpansions == 0:
            expanAccuracy = 0.0
        else:
            expanAccuracy = sumCorrectExpansions / sumExpansions

        logger.critical(
            "Correct Expansions %d of %d, accuracy: %f",
            sumCorrectExpansions,
            sumExpansions,
            expanAccuracy,
        )

        incorrect_missing_expansion = [
            item["incorrect_missing_expansion"]
            for item in self.cumulativeResults.values()
            if item is not None
        ]
        total_incorrect_missing_expansion = sum(incorrect_missing_expansion)
        avg_incorrect_missing_expansion = mean(incorrect_missing_expansion)

        logger.critical(
            "Incorrect missing expansions Mean %f, total: %f",
            avg_incorrect_missing_expansion,
            total_incorrect_missing_expansion,
        )

        total_poss_expansions = [
            item["total_expansions"] - item["incorrect_missing_expansion"]
            for item in self.cumulativeResults.values()
            if item is not None
        ]

        sumPossExpansions = sum(total_poss_expansions)
        if sumPossExpansions == 0:
            expanPossAccuracy = 0.0
        else:
            expanPossAccuracy = sumCorrectExpansions / sumPossExpansions
        logger.critical(
            "Correct Possible Expansions %d of %d, accuracy: %f",
            sumCorrectExpansions,
            sumPossExpansions,
            expanPossAccuracy,
        )

        possExpanAccPerFold = [
            item["average_possible_correct_expansions"]
            for item in self.cumulativeResults.values()
            if item is not None
        ]

        mean_poss_val_expan = mean(possExpanAccPerFold)
        std_poss_dev_expan = std(possExpanAccPerFold)
        logger.critical(
            "Possible Expansions Mean %f, Standard Deviation: %f",
            mean_poss_val_expan,
            std_poss_dev_expan,
        )

        validDisamSuccesses = [
            item["correct_disambiguations"]
            for item in self.cumulativeResults.values()
            if item is not None
        ]

        sumDisamSuccesses = sum(validDisamSuccesses)

        total_ambiguous_acronyms = [
            item["total_disambiguations"]
            for item in self.cumulativeResults.values()
            if item is not None
        ]

        sumAmbiguous = sum(total_ambiguous_acronyms)

        if sumAmbiguous == 0:
            disamAccuracy = 0.0
        else:
            disamAccuracy = sumDisamSuccesses / sumAmbiguous

        logger.critical(
            "Correct Disambiguations %d of %d, accuracy: %f",
            sumDisamSuccesses,
            sumAmbiguous,
            disamAccuracy,
        )

        disamAccPerFold = [
            item["correct_disambiguations"] / item["total_disambiguations"]
            if item is not None and item["total_disambiguations"] > 0
            else 0
            for item in self.cumulativeResults.values()
        ]

        mean_val_disam = mean(disamAccPerFold)
        std_dev_disam = std(disamAccPerFold)
        logger.critical(
            "Disambiguation Mean %f, Standard Deviation: %f",
            mean_val_disam,
            std_dev_disam,
        )

        incorrect_missing_labels = [
            item["incorrect_missing_label"]
            for item in self.cumulativeResults.values()
            if item is not None
        ]
        total_incorrect_missing_labels = sum(incorrect_missing_labels)
        avg_incorrect_missing_labels = mean(incorrect_missing_labels)

        logger.critical(
            "Incorrect missing labels Mean %f, total: %f",
            avg_incorrect_missing_labels,
            total_incorrect_missing_labels,
        )

        # Possible Disambiguations count, counts only the disambiguations that have the true label in the training set
        total_poss_ambiguous_acronyms = [
            item["total_disambiguations"] - item["incorrect_missing_label"]
            for item in self.cumulativeResults.values()
            if item != None
        ]

        sumPossAmbiguous = sum(total_poss_ambiguous_acronyms)
        if sumPossAmbiguous == 0:
            disamPossAccuracy = 0.0
        else:
            disamPossAccuracy = sumDisamSuccesses / sumPossAmbiguous
        logger.critical(
            "Correct Possible Disambiguations %d of %d, accuracy: %f",
            sumDisamSuccesses,
            sumPossAmbiguous,
            disamPossAccuracy,
        )

        possDisamAccPerFold = [
            item["correct_disambiguations"]
            / (item["total_disambiguations"] - item["incorrect_missing_label"])
            if item != None
            and (item["total_disambiguations"] - item["incorrect_missing_label"]) > 0
            else 0
            for item in self.cumulativeResults.values()
        ]

        mean_poss_val_disam = mean(possDisamAccPerFold)
        std_poss_dev_disam = std(possDisamAccPerFold)
        logger.critical(
            "Possible Disambiguation Mean %f, Standard Deviation: %f",
            mean_poss_val_disam,
            std_poss_dev_disam,
        )

        macro_prec = mean(
            [item["macro_prec"] for item in self.cumulativeResults.values()]
        )
        macro_recall = mean(
            [item["macro_recall"] for item in self.cumulativeResults.values()]
        )
        macro_f1_measure = mean(
            [item["macro_f1_measure"] for item in self.cumulativeResults.values()]
        )
        logger.critical(
            "Macro Averages Precision: %f, Recall: %f, F1-Measure: %f",
            macro_prec,
            macro_recall,
            macro_f1_measure,
        )

        macro_prec_amgiguous = mean(
            [
                item["macro_prec_ambiguous_acro_exp_in_db"]
                for item in self.cumulativeResults.values()
            ]
        )
        macro_recall_amgiguous = mean(
            [
                item["macro_recall_ambiguous_acro_exp_in_db"]
                for item in self.cumulativeResults.values()
            ]
        )
        macro_f1_measure_amgiguous = mean(
            [
                item["macro_f1_measure_ambiguous_acro_exp_in_db"]
                for item in self.cumulativeResults.values()
            ]
        )
        logger.critical(
            "Ambiguous Acronym Expansion in DB macro Averages Precision: %f, Recall: %f, F1-Measure: %f",
            macro_prec_amgiguous,
            macro_recall_amgiguous,
            macro_f1_measure_amgiguous,
        )
        measure_values = [
            expanAccuracy,
            mean_val,
            std_dev,
            expanPossAccuracy,
            mean_poss_val_expan,
            std_poss_dev_expan,
            disamAccuracy,
            mean_val_disam,
            std_dev_disam,
            disamPossAccuracy,
            mean_poss_val_disam,
            std_poss_dev_disam,
            macro_prec,
            macro_recall,
            macro_f1_measure,
            macro_prec_amgiguous,
            macro_recall_amgiguous,
            macro_f1_measure_amgiguous,
        ]

        return dict(zip(QUALITY_FIELDS_LIST, measure_values))
