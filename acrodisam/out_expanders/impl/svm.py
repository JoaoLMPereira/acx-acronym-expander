"""
Created on Aug 29, 2018

@author: jpereira
"""
from enum import Enum
import numpy as np
from sklearn.model_selection import LeaveOneOut
from collections import Counter

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from .._base import (
    OutExpanderWithTextRepresentator,
    OutExpanderWithTextRepresentatorFactory,
)

from pydantic import BaseModel, validate_arguments, PositiveInt
from typing import List, Optional
from pydantic.types import PositiveFloat
from inputters import TrainOutDataManager


def oversampling(x, y):
    counts = Counter(y)
    for e, c in counts.items():
        if c < 2:
            indx = y.index(e)
            y.append(e)
            x.append(x[indx])
    return x, y


class FactoryExpanderSVM(OutExpanderWithTextRepresentatorFactory):
    class lossEnum(str, Enum):
        l1 = "l1"
        l2 = "l2"

    @validate_arguments
    def __init__(
        self,
        loss: lossEnum = "l1",
        c: PositiveFloat = 0.1,
        balance_class_weights: bool = False,
        *args,
        **kwargs
    ):
        '''
        argparser = self._create_argparse()
        argparser.add_argument("penalty", default='l2', choices=['l1', 'l2'], help="""Specifies the norm used in the penalization.
        The 'l2' penalty is the standard used in SVC. The 'l1' leads to ``coef_``vectors that are sparse.""")
        argparser.add_argument("c", type=float, default=1.0, help="""Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.""")

        super().__init__(*args, argparser=argparser, **kwargs)
        self.loss = self.namespace.penalty
        self.c = self.namespace.c
        '''
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.c = c
        self.balance_class_weights = balance_class_weights

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        text_representator = self._get_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )
        return _ExpanderSVM(
            self.loss, self.c, self.balance_class_weights, text_representator
        )


class _ExpanderSVM(OutExpanderWithTextRepresentator):
    def __init__(self, loss, c, balance_class_weights, text_representator):
        super().__init__(text_representator)
        self.loss = loss
        self.c = c
        self.class_weight = "balanced" if balance_class_weights else None
        self.classifier = None

    def fit(self, X_train, y_train):
        if self.loss == "l1":
            dual = False
            loss = "squared_hinge"
        else:
            dual = True
            loss = "squared_hinge"

        self.classifier = LinearSVC(
            C=self.c,
            penalty=self.loss,
            loss=loss,
            dual=dual,
            class_weight=self.class_weight,
        )  # , max_iter= 10000)
        x_train_transf = np.ascontiguousarray(list(X_train))
        self.classifier.fit(x_train_transf, list(y_train))
        # self.classifier = LinearSVC(C=self.c, loss=self.loss)
        # kf = StratifiedKFold(n_splits=2)
        """
        cv = LeaveOneOut()
        self.classifier = CalibratedClassifierCV(self.classifier, method='sigmoid', cv=cv)
        self.classifier.fit(*oversampling(list(X_train), list(y_train)))
        """

    def predict(self, X_test, acronym):
        if isinstance(X_test, np.ndarray):
            X_test = X_test.reshape(1, -1,order= 'c')
        else:
            aux = list()
            aux.append(X_test)
            X_test = aux
        labels = self.classifier.predict(X_test)

        decisions = self.classifier.decision_function(X_test)

        confidences = self._get_confidences_from_decision_function(labels, decisions)
        return labels, confidences
    
    def predict_confidences(self, text_representation, acronym, distinct_expansions):
        confidences_dict = {}
        decision = self.classifier.decision_function(text_representation.reshape(1, -1))[0]
        if hasattr(decision, "__iter__"):
            for exp, conf in zip(self.classifier.classes_, decision):
                confidences_dict[exp] = conf
        else:
            #binary case
            if decision > 0:
                confidences_dict[self.classifier.classes_[1]] = decision
                confidences_dict[self.classifier.classes_[0]] = 0
            else:
                confidences_dict[self.classifier.classes_[1]] = 0
                confidences_dict[self.classifier.classes_[0]] = abs(decision)
        
        return confidences_dict
