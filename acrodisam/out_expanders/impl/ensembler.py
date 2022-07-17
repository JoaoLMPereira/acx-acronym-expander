"""
Selects the expansion whose vector obtains the closer cosine distance to the test instance
"""
import csv
import json
import operator
import os
import re

from sklearn.ensemble import GradientBoostingClassifier

from helper import grouped_longest
from inputters import TrainOutDataManager
import numpy as np
from out_expanders import get_out_expander_factory
from out_expanders._base import OutExpanderArticleInput
from run_config import RunConfig
from string_constants import FOLDER_LOGS

from .._base import OutExpander, OutExpanderFactory
from abc import abstractmethod


def get_cv_expansion(experiment_name):
    exp_per_fold_doc_acro = {}
    with open(
        FOLDER_LOGS + "expansions_" + experiment_name + "_confidences_CV.csv", "r"
    ) as f:
        reader = csv.DictReader(
            f,
        )
        for row in reader:
            fold = row["fold"]
            doc_id = row["doc_id"]
            acronym = row["acronym"]
            dict_key = (fold, doc_id, acronym)
            exp_per_fold_doc_acro[dict_key] = row["expansion"]

    return exp_per_fold_doc_acro


def build_train_data(experiment_name):
    confidences_dict = {}
    with open(
        FOLDER_LOGS + "confidences_" + experiment_name + "_confidences_CV.csv", "r"
    ) as f:
        reader = csv.DictReader(
            f,
        )

        for row in reader:

            fold = row["fold"]
            doc_id = row["doc_id"]
            acronym = row["acronym"]

            confidences_values = confidences_dict.setdefault(
                (fold, doc_id, acronym), {}
            )

            confidences_values[
                (row["out_expander"], row["out_expander_args"])
            ] = json.loads(row["confidences_json_dict"])

    train_x = []
    train_y = []

    out_expanders = list(next(iter(confidences_dict.values())).keys())

    exp_per_fold_doc_acro = get_cv_expansion(experiment_name)

    for k, confidences in confidences_dict.items():
        true_expansion = exp_per_fold_doc_acro.get(k)

        expansion_confidences = {}
        for out_expander in out_expanders:
            for expansion, conf_value in confidences[out_expander].items():
                confidence_values = expansion_confidences.setdefault(expansion, [])
                confidence_values.append(conf_value)

        for expansion, confidence_values in expansion_confidences.items():
            train_x.append(confidence_values)
            y = 1 if true_expansion == expansion else 0
            train_y.append(y)

    return train_x, train_y, out_expanders

def get_expanders(datasetname: str):
        
    expanders = []
    for file in os.listdir(FOLDER_LOGS):
        m = re.match('^confidences_'+datasetname+'_([.:=_\-\\w]+).csv$', file)
        if m:
            expander = m.group(1)
            if "ensembler" not in expander:
                expanders.append(expander)
    return expanders

def get_confidences(experiment_name, fold):
    if not experiment_name.endswith("_confidences"):
        experiment_name += "_confidences"
    confidences_dict = {}
    expanders = get_expanders(experiment_name)

    for expander in expanders:
        with open(FOLDER_LOGS + "confidences_" + experiment_name + "_"+expander+".csv", "r") as f:
            reader = csv.DictReader(f)
    
            for row in reader:
                fold = row["fold"]
                if fold != row["fold"]:
                    continue
    
                doc_id = row["doc_id"]
                acronym = row["acronym"]
    
                confidences_values = confidences_dict.setdefault((doc_id, acronym), [])
    
                confidences_values.append(
                    (
                        expander,
                        json.loads(row["confidences_json_dict"]),
                    )
                )

    return confidences_dict


class FactoryExpanderEnsembler(OutExpanderFactory):

    def __init__(
        self,
        mode="",
        weights=None,
        run_config: RunConfig = RunConfig(),
        **kwargs
    ):
        """
        """
        self.weights = weights
        self.out_expander_factories = []

        self.experiment_name = run_config.name

        self.mode = mode

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):

        if (
            train_data_manager.get_fold() is not None
            and train_data_manager.get_fold() != "TrainData"
        ):
            raise NotImplementedError(
                "This out-expander uses its own cross-validation. Not accepting fold: %s"
                % train_data_manager.get_fold()
            )

        confidences_dict = get_confidences(
            self.experiment_name, train_data_manager.get_fold()
        )
        if self.mode != "hard" and self.mode != "soft" and self.mode != "threshold":
            train_x, train_y, out_expanders = build_train_data(self.experiment_name)
            # TODO add other models
            model = (
                GradientBoostingClassifier(
                    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
                )
                .fit(train_x, train_y)
                .predict_proba
            )
            return _TrainedEnsembler(confidences_dict, model, out_expanders)

        if self.mode == "hard":
            return _HardVoteEnsembler(confidences_dict)
        
        if self.mode == "threshold":
            return _ThresholdHardVoteEnsembler(confidences_dict)

        return _SoftVoteEnsembler(confidences_dict)


class _VoteEnsembler(OutExpander):

    def __init__(self, confidences_dict, weights=None):
        self.weights = weights
        self.confidences_dict = confidences_dict
        
    @abstractmethod
    def get_confidences(self, doc_id, acronym):
        pass

    def predict(self, doc_id, acronym):

        soft_voting = self.get_confidences(doc_id,acronym)
        if not soft_voting:
            return "None", 0.0

        most_voted_exp = max(soft_voting.items(), key=operator.itemgetter(1))[0]

        return most_voted_exp, soft_voting[most_voted_exp]


    def process_article(self, out_expander_input: OutExpanderArticleInput):
        predicted_expansions = []
        doc_id = out_expander_input.test_article_id
        for acronym in out_expander_input.acronyms_list:
            exp, conf = self.predict(doc_id, acronym)
            predicted_expansions.append((exp, conf))

        return predicted_expansions

    def process_article_return_confidences(self, out_expander_input: OutExpanderArticleInput):
        confidences_list = []
        doc_id = out_expander_input.test_article_id
        for acronym, distinct_expansions in zip(out_expander_input.acronyms_list, out_expander_input.distinct_expansions_list):
            all_zeros_exp_dict = {exp:0.0 for exp in distinct_expansions}
            confidences_dict = self.get_confidences(doc_id, acronym)
            if not confidences_dict:
                confidences_dict = {exp:0.0 for exp in distinct_expansions}
            else:
                confidences_dict = {**all_zeros_exp_dict, **confidences_dict} # Doing this all exp are represented in the dict
            confidences_list.append(confidences_dict)
        return confidences_list


class _HardVoteEnsembler(_VoteEnsembler):

    def get_confidences(self, doc_id, acronym):
        confidences = self.confidences_dict.get((doc_id, acronym))
        if not confidences:
            return None
        hard_voting = {}

        for item in confidences:
            confidences_exp_dict = item[1]
            predict_exp = max(confidences_exp_dict.items(), key=operator.itemgetter(1))[
                0
            ]

            votes = hard_voting.get(predict_exp, 0)
            hard_voting[predict_exp] = votes + 1

        return hard_voting


class _SoftVoteEnsembler(_VoteEnsembler):
    def normalize_conf_values(self, conf_dict):

        values = list(conf_dict.values())
        min_value = min(values)
        if min_value < 0:  # negative values e.g., SVM decision func values
            non_neg_min = abs(min_value)
            conf_dict = {k: v + non_neg_min for (k, v) in conf_dict.items()}

        values_sum = sum(conf_dict.values())
        for k, v in conf_dict.items():
            yield k, (v / values_sum) if values_sum != 0 else 0
            
    def get_confidences(self, doc_id, acronym):
        confidences = self.confidences_dict.get((doc_id, acronym))
        if not confidences:
            return None
        soft_voting = {}

        for item in confidences:
            confidences_exp_dict = item[1]
            for exp, conf_value in self.normalize_conf_values(confidences_exp_dict):
                soft_votes = soft_voting.get(exp, 0.0)
                soft_voting[exp] = soft_votes + conf_value
                
        return soft_voting
    
    
class _ThresholdHardVoteEnsembler(_VoteEnsembler):
    def normalize_conf_values(self, conf_dict):

        values = list(conf_dict.values())
        min_value = min(values)
        if min_value < 0:  # negative values e.g., SVM decision func values
            non_neg_min = abs(min_value)
            conf_dict = {k: v + non_neg_min for (k, v) in conf_dict.items()}

        values_sum = sum(conf_dict.values())
        for k, v in conf_dict.items():
            yield k, (v / values_sum) if values_sum != 0 else 0
    
    def get_predict(self, expander, confidences_exp_dict):
        predict_exp, confidence = max(confidences_exp_dict.items(), key=operator.itemgetter(1))
        
        if expander.startswith("svm"):
            if confidence >= 0.25:
                return predict_exp
        elif expander.startswith("cossim_tfidf"):
            if confidence >= 0.10:
                return predict_exp
        elif expander == "cossim_classic_context_vector":
            if confidence >= 0.15:
                return predict_exp
        elif expander.startswith("sci_dr"):
            return predict_exp
        elif expander.startswith("cossim_doc2vec") or expander.startswith("cossim_sbert"):
            if confidence >= 0.5:
                confidences_exp_dict.pop(predict_exp)
                second_confidence = max(confidences_exp_dict.items(), key=operator.itemgetter(1))[1]
                if second_confidence > 0 and confidence / second_confidence >= 1.15:
                    return predict_exp
            
        return None
    
    def get_confidences(self, doc_id, acronym):
        confidences = self.confidences_dict.get((doc_id, acronym))
        if not confidences:
            return None
        hard_voting = {}

        for item in confidences:
            expander = item[0]
            confidences_exp_dict = item[1]
            predict_exp = self.get_predict(expander, confidences_exp_dict)
            
            if not predict_exp:
                continue
            
            votes = hard_voting.get(predict_exp, 0)
            hard_voting[predict_exp] = votes + 1

        return hard_voting
    
class _TrainedEnsembler(OutExpander):

    def __init__(self, confidences_dict, model, out_expanders):
        self.confidences_dict = confidences_dict
        self.model = model
        self.out_expanders = out_expanders

    def normalize_conf_values(self, conf_dict):

        values = list(conf_dict.values())
        min_value = min(values)
        if min_value < 0:  # negative values e.g., SVM decision func values
            non_neg_min = abs(min_value)
            conf_dict = {k: v + non_neg_min for (k, v) in conf_dict.items()}

        values_sum = sum(conf_dict.values())
        for k, v in conf_dict.items():
            yield k, v / values_sum

    def predict(self, doc_id, acronym):

        confidences = self.confidences_dict.get((doc_id, acronym))

        expanders_confidences = {(item[0]): item[1] for item in confidences}

        expansion_confidences = {}
        for out_expander in self.out_expanders:
            for expansion, conf_value in expanders_confidences[out_expander].items():
                confidence_values = expansion_confidences.setdefault(expansion, [])
                confidence_values.append(conf_value)

        test_x = []
        test_y = []
        for expansion, confidence_values in expansion_confidences.items():
            test_x.append(confidence_values)
            test_y.append(expansion)

        predictions = self.model(test_x)

        highest_pred_idx = np.argmax(predictions[:, 1])

        return test_y[highest_pred_idx], predictions[:, 1][highest_pred_idx]

    def process_article(self, out_expander_input: OutExpanderArticleInput):
        predicted_expansions = []
        doc_id = out_expander_input.test_article_id
        for acronym in out_expander_input.acronyms_list:
            exp, conf = self.predict(doc_id, acronym)
            predicted_expansions.append((exp, conf))

        return predicted_expansions
