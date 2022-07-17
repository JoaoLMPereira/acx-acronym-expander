"""
Random Forest

Created on Aug 29, 2018

@author: jpereira
"""
from enum import Enum
import math

from pydantic import validate_arguments, PositiveInt
from sklearn.ensemble import RandomForestClassifier

from inputters import TrainOutDataManager

from .._base import (
    OutExpanderWithTextRepresentator,
    OutExpanderWithTextRepresentatorFactory,
)


class FactoryExpanderRF(OutExpanderWithTextRepresentatorFactory):
    class MaxFeaturesEnum(str, Enum):
        none = "None"
        root3 = "root3"
        root4 = "root4"
        sqrt = "sqrt"
        log2 = "log2"

    @validate_arguments
    def __init__(
        self,
        n_estimators: PositiveInt = 100,
        max_features: MaxFeaturesEnum = "None",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        text_representator = self._get_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )
        return _ExpanderRF(self.n_estimators, self.max_features, text_representator)


class _ExpanderRF(OutExpanderWithTextRepresentator):
    def __init__(self, n_estimators, max_features, text_representator):
        super().__init__(text_representator)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.classifier = None

    def fit(self, X_train, y_train):
        X_train = list(X_train)
        y_train = list(y_train)
        if self.max_features == "None":
            self.max_features = None
        elif self.max_features == "root3":
            if isinstance(X_train, list):
                n_features = X_train[0].shape[0]
            else:
                n_features = X_train.shape[1]
            self.max_features = int(round(math.pow(n_features, 1 / 3)))
        elif self.max_features == "root4":
            if isinstance(X_train, list):
                n_features = X_train[0].shape[0]
            else:
                n_features = X_train.shape[1]
            self.max_features = int(round(math.pow(n_features, 1 / 4)))

        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=123456,
        )
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test, acronym):
        proba = self.classifier.predict_proba(X_test.reshape(1, -1))

        labels, confidences = self._get_labels_and_confidences_from_proba(proba)

        return labels, confidences
    
    def predict_confidences(self, text_representation, acronym, distinct_expansions):
        confidences_dict = {}
        prob_classes = self.classifier.predict_proba(text_representation.reshape(1, -1))[0]
        
        for exp, conf in zip(self.classifier.classes_, prob_classes):
            confidences_dict[exp] = conf
        
        return confidences_dict
