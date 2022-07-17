"""
Logistic Regression
Created on Aug 29, 2018

@author: jpereira
"""
from sklearn.linear_model import LogisticRegression
from enum import Enum

from .._base import (
    OutExpanderWithTextRepresentator,
    OutExpanderWithTextRepresentatorFactory,
)
from pydantic import BaseModel, validate_arguments, PositiveInt
from typing import List, Optional
from pydantic.types import PositiveFloat
from inputters import TrainOutDataManager


class FactoryExpanderLR(OutExpanderWithTextRepresentatorFactory):
    class lossEnum(str, Enum):
        l1 = "l1"
        l2 = "l2"

    @validate_arguments
    def __init__(self, loss: lossEnum = "l1", c: PositiveFloat = 0.1, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.c = c
        # self.balance_class_weights = balance_class_weights

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        text_representator = self._get_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )
        return ExpanderLR(self.loss, self.c, text_representator)


class ExpanderLR(OutExpanderWithTextRepresentator):
    def __init__(self, loss, c, text_representator):
        super().__init__(text_representator)
        self.loss = loss
        self.c = c
        self.classifier = None

    def fit(self, X_train, y_train):
        self.classifier = LogisticRegression(
            C=self.c, penalty=self.loss, solver="liblinear", multi_class="ovr"
        )
        # self.classifier.solver
        self.classifier.fit(list(X_train), list(y_train))

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
