"""Decision-threshold wrapper used by saved classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Wrap any predict_proba model with a fixed decision threshold."""

    def __init__(self, estimator=None, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
