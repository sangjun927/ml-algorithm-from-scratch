import numpy as np
from sklearn.utils import resample
from collections import Counter

from dtree_rf import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False, max_features=0.3, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.trees = []
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples, self.n_estimators)) if self.oob_score else None
        oob_counts = np.zeros(n_samples) if self.oob_score else None

        for i in range(self.n_estimators):
            bootstrap_sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_sample_indices, assume_unique=True)

            tree = ClassifierTree621(min_samples_leaf=self.min_samples_leaf, max_features=self.max_features) if isinstance(self, RandomForestClassifier621) else RegressionTree621(min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)

            tree.fit(X[bootstrap_sample_indices], y[bootstrap_sample_indices])
            self.trees.append(tree)

            if self.oob_score:
                oob_pred = tree.predict(X[oob_indices])
                oob_predictions[oob_indices, i] = oob_pred
                oob_counts[oob_indices] += 1

        if self.oob_score:
            valid_oob = oob_counts > 0
            oob_avg_predictions = np.sum(oob_predictions, axis=1) / np.where(oob_counts == 0, 1, oob_counts)
            self.oob_score_ = accuracy_score(y[valid_oob], np.round(oob_avg_predictions[valid_oob])) if isinstance(self, RandomForestClassifier621) else r2_score(y[valid_oob], oob_avg_predictions[valid_oob])

            
class RandomForestRegressor621(RandomForest621):
    
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score, max_features=max_features, min_samples_leaf=min_samples_leaf)
        
        self.trees = []
        self.max_features = max_features

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions = np.array([tree.predict(X_test) for tree in self.trees])
        return np.mean(predictions, axis=0)

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        predicted_y = self.predict(X_test)
        return r2_score(y_test, predicted_y)


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.max_features = max_features

    def predict(self, X_test) -> np.ndarray:
        all_predictions = np.array([tree.predict(X_test) for tree in self.trees])
        final_predictions = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=all_predictions)
        return final_predictions.astype(int)
    

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        predicted_y = self.predict(X_test)
        return accuracy_score(y_test, predicted_y)
