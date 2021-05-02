from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
import numpy as np
from logzero import logger

class SVCWrapper(BaseEstimator):
    def __init__(self, probability: bool, kernel: str='linear', C: float=1.0, gamma: float=None):
        self.probability = probability
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

        if kernel == 'linear':
            # model = LinearSVC(random_state=42, class_weight='balanced', verbose=1, C=C)
            # self.model = CalibratedClassifierCV(model) if self.probability else model
            self.model = SVC(random_state=42, class_weight='balanced', verbose=1, kernel=kernel, C=C, probability=probability)
        else:
            self.model = SVC(random_state=42, class_weight='balanced', verbose=1, kernel=kernel, C=C, gamma=gamma, probability=probability)

    def fit(self, X, y, sample_weight=None):
        X = np.array(X)
        y = np.array(y)
        class_ = set(np.unique(y))
        logger.info("y {}".format(class_))
        assert len(class_) > 1
        # min_num = 1000000
        # for c in class_:
        #     indices = np.where(y==c)[0]
        #     logger.info("{} {} {}".format(c, indices, len(indices)))
        #     classes[c] = indices
        #     if len(indices) < min_num:
        #         min_num = len(indices)
        # X_ = []
        # y_ = []
        # for c in class_:
        #     logger.info("{} {}".format(classes[c], min_num))
        #     indices = np.random.choice(classes[c], min_num, replace=False)
        #     X_.extend(X[indices])
        #     y_.extend(y[indices])

        # X = np.array(X_)
        # y = np.array(y_)
        # logger.info("{} {}".format(X.shape, np.unique(y, return_counts=True)))

        self.model.fit(X, y, sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        assert self.probability
        return self.model.predict_proba(X)


    def decision_function(self, X):
        return self.model.decision_function(X)
