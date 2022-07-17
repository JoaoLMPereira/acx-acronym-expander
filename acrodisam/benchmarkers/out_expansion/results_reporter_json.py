"""
Needs refactoring

Created on Feb 20, 2020

@author: jpereira
"""
from collections import defaultdict
import json

from numpy import mean, std

from Logger import logging
from benchmarkers.results_reporter_base import ResultsReporterBase
from helper import AcronymExpansion, flatten_to_str_list
from string_constants import (
    FILE_BENCHMARK_RESULTS,
    FILE_REPORT_OUT_EXPANSION,
    FILE_BENCHMARK_RESULTS_JSON,
)


logger = logging.getLogger(__name__)


def run_evaluation(pred_list, verbose=True):

    gold = dict([(d["id"], d["expansion"]) for d in pred_list])

    pred = dict([(d["id"], d["prediction"]) for d in pred_list])
    pred = [pred.get(k) for k, v in gold.items()]
    gold = [gold.get(k) for k, v in gold.items()]
    p, r, f1 = score_expansion(gold, pred, verbos=verbose)
    return p, r, f1


def score_expansion(key, prediction, verbos=False):
    correct = 0
    for i in range(len(key)):
        if key[i] == prediction[i]:
            correct += 1
    acc = correct / len(prediction)

    expansions = set()

    correct_per_expansion = defaultdict(int)
    total_per_expansion = defaultdict(int)
    pred_per_expansion = defaultdict(int)
    for i in range(len(key)):
        expansions.add(key[i])
        total_per_expansion[key[i]] += 1
        pred_per_expansion[prediction[i]] += 1
        if key[i] == prediction[i]:
            correct_per_expansion[key[i]] += 1

    precs = defaultdict(int)
    recalls = defaultdict(int)

    for exp in expansions:
        precs[exp] = (
            correct_per_expansion[exp] / pred_per_expansion[exp]
            if exp in pred_per_expansion
            else 1
        )
        recalls[exp] = correct_per_expansion[exp] / total_per_expansion[exp]

    micro_prec = sum(correct_per_expansion.values()) / sum(pred_per_expansion.values())
    micro_recall = sum(correct_per_expansion.values()) / sum(
        total_per_expansion.values()
    )
    micro_f1 = (
        2 * micro_prec * micro_recall / (micro_prec + micro_recall)
        if micro_prec + micro_recall != 0
        else 0
    )

    macro_prec = sum(precs.values()) / len(precs)
    macro_recall = sum(recalls.values()) / len(recalls)
    macro_f1 = (
        2 * macro_prec * macro_recall / (macro_prec + macro_recall)
        if macro_prec + macro_recall != 0
        else 0
    )

    if verbos:
        logger.critical("Accuracy: {:.3%}".format(acc))
        logger.critical("-" * 10)
        logger.critical("Micro Precision: {:.3%}".format(micro_prec))
        logger.critical("Micro Recall: {:.3%}".format(micro_recall))
        logger.critical("Micro F1: {:.3%}".format(micro_f1))
        logger.critical("-" * 10)
        logger.critical("Macro Precision: {:.3%}".format(macro_prec))
        logger.critical("Macro Recall: {:.3%}".format(macro_recall))
        logger.critical("Macro F1: {:.3%}".format(macro_f1))
        logger.critical("-" * 10)

    return macro_prec, macro_recall, macro_f1


class ResultsReporter(ResultsReporterBase):
    """
    classdocs
    """

    def __init__(
        self,
        experiment_name: str,
        experiment_parameters,
        report_file_name: str = FILE_REPORT_OUT_EXPANSION,
        save_results_to_file: bool = False,
        save_predictions_to_json=True,
    ):
        """
        Constructor
        """

        results_file_name = str(FILE_BENCHMARK_RESULTS).format(
            experiment_name + "_" + "_".join(flatten_to_str_list(experiment_parameters))
        )
        self.results_file_name_json = str(FILE_BENCHMARK_RESULTS_JSON).format(
            experiment_name + "_" + "_".join(flatten_to_str_list(experiment_parameters))
        )
        cummulative_results_init = {
            "correct_expansions": 0,
            "incorrect_expansions": 0,
            "incorrect_missing_expansion": 0,
            "correct_disambiguations": 0,
            "incorrect_disambiguations": 0,
            "incorrect_missing_label": 0,
        }

        super().__init__(
            experiment_name,
            experiment_parameters,
            report_file_name,
            save_results_to_file,
            results_file_name,
            cummulative_results_init,
        )

        self.save_predictions_to_json = save_predictions_to_json

    def __enter__(self):
        if self.save_predictions_to_json:
            self.predictions_list = []

        return super().__enter__()

    def _computeResults(self):
        if self.save_predictions_to_json:
            only_predictions_list = [
                {"id": item["id"], "prediction": item["prediction"]}
                for item in self.predictions_list
            ]
            with open(self.results_file_name_json, "w") as fp:
                json.dump(only_predictions_list, fp)

        super()._computeResults()

    # def __exit__(self, exc_type, exc_value, traceback):
    #    super().__exit__(exc_type, exc_value, traceback)

    def plotStats(self):
        list_metrics = super().plotStats()

        macro_p, macro_r, macro_f1 = run_evaluation(self.predictions_list)

        return list_metrics + [macro_p, macro_r, macro_f1]

    def _simExpansionExists(self, testExp, expList):
        for candidateExp in expList:
            if AcronymExpansion.areExpansionsSimilar(testExp, candidateExp):
                return True

        return False

    def _process_quality_results(
        self,
        fold_cummulative_results,
        results_writer,
        docId,
        actual_expansions,
        predicted_expansions,
    ):

        for acronym in actual_expansions:
            actual_expansion = actual_expansions[acronym]

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
                    ";".join(predicted_expansions[acronym].options),
                )

                if results_writer:
                    results_writer(
                        acronym,
                        actual_expansion,
                        predicted_expansions[acronym].expansion,
                        predicted_expansions[acronym].confidence,
                        True,
                    )

                fold_cummulative_results["correct_expansions"] += 1
                if len(predicted_expansions[acronym].options) > 1:
                    fold_cummulative_results["correct_disambiguations"] += 1
                    # if results_writer:
                    #    results_writer(acronym, actual_expansion, predicted_expansions[acronym][0].expansion, predicted_expansions[acronym][0].confidence, True)
            else:

                if acronym in predicted_expansions:
                    predicted_expansion = predicted_expansions[acronym].expansion
                    confidence = predicted_expansions[acronym].confidence
                    if results_writer:
                        results_writer(
                            acronym,
                            actual_expansion,
                            predicted_expansions[acronym].expansion,
                            confidence,
                            False,
                        )
                    fold_cummulative_results["incorrect_expansions"] += 1

                    if len(predicted_expansions[acronym].options) > 1:
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
                            # if results_writer:
                            #    results_writer(acronym, actual_expansion, predicted_expansions[acronym][0].expansion, confidence, False)
                            pass
                    else:
                        if (
                            self._simExpansionExists(
                                actual_expansion, predicted_expansions[acronym].options
                            )
                            == False
                        ):
                            fold_cummulative_results["incorrect_missing_expansion"] += 1

                    logger.debug(
                        "Expansion matching failed (%s): %s, %s, confidence: %f, options: %r",
                        acronym,
                        actual_expansion,
                        predicted_expansion,
                        confidence,
                        ";".join(predicted_expansions[acronym].options),
                    )

                elif acronym in predicted_expansions:
                    logger.debug(
                        "Expansion matching failed (%s): %s, no option available",
                        acronym,
                        actual_expansion,
                    )
                    if results_writer:
                        results_writer(acronym, actual_expansion, None, 0, False)
                    fold_cummulative_results["incorrect_expansions"] += 1
                    fold_cummulative_results["incorrect_missing_expansion"] += 1

                else:
                    logger.error("Expansion not predicted for %s", acronym)

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
        else:
            fold_results["average_correct_expansions"] = 0
            fold_results["average_incorrect_expansions"] = 0

        if (totalExpansions - incorrect_missing_expansion) > 0:
            fold_results["average_possible_correct_expansions"] = fold_results[
                "correct_expansions"
            ] / (totalExpansions - incorrect_missing_expansion)
        else:
            fold_results["average_possible_correct_expansions"] = 0

        fold_results["total_disambiguations"] = totalDisambiguations

    def _writeResult(
        self,
        fold,
        docId,
        acronym,
        actual_expansion,
        predicted_expansion,
        confidence,
        success,
    ):
        if self.saveResultsToFile:
            self.resultsFileWriter.writerow(
                [
                    fold,
                    docId,
                    acronym,
                    actual_expansion,
                    predicted_expansion,
                    confidence,
                    success,
                ]
            )

        if self.save_predictions_to_json:
            self.predictions_list.append(
                {
                    "id": docId,
                    "prediction": predicted_expansion,
                    "expansion": actual_expansion,
                }
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
            if item != None
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
            if item != None
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
            if item != None
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
            if item != None
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
            if item != None
        ]

        sumDisamSuccesses = sum(validDisamSuccesses)

        total_ambiguous_acronyms = [
            item["total_disambiguations"]
            for item in self.cumulativeResults.values()
            if item != None
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
            if item != None
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

        return [
            mean_val,
            std_dev,
            expanAccuracy,
            mean_poss_val_expan,
            std_poss_dev_expan,
            expanPossAccuracy,
            disamAccuracy,
            mean_val_disam,
            std_dev_disam,
            disamPossAccuracy,
            mean_poss_val_disam,
            std_poss_dev_disam,
        ]
