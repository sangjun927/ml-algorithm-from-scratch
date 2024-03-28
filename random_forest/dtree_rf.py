import numpy as np
from scipy import stats
from collections import Counter

from sklearn.metrics import r2_score, accuracy_score
class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test.T[self.col] < self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)
        
    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node. This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] < self.split:
            return self.lchild.leaf(x_test) if isinstance(self.lchild, DecisionNode) else self.lchild
        else:
            return self.rchild.leaf(x_test) if isinstance(self.rchild, DecisionNode) else self.rchild
        
class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        # self.y = y 

    def predict(self, x_test):
        return self.prediction



def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    counter = Counter(x)
    total = len(x)
    gini_score = 1 - sum((count / total) ** 2 for count in counter.values())
    return gini_score


def find_best_split(X, y, loss, min_samples_leaf, max_features):
    n_samples, n_features = X.shape
    best_loss = float("inf")
    best_col = -1
    best_split = None

    # pre-compute the loss for the whole node to avoid recalculating it
    current_loss = loss(y)

    # randomly select a subset of features if max_features < 1.0
    if max_features < 1.0:
        n_selected_features = int(n_features * max_features)
        features = np.random.choice(n_features, n_selected_features, replace=False)
    else:
        features = range(n_features)

    for col in features:
        # set()
        unique_values = set(X[:, col])

        for split in unique_values:
            left_mask = X[:, col] < split
            right_mask = ~left_mask

            if np.sum(left_mask) < min_samples_leaf or np.sum(right_mask) < min_samples_leaf:
                continue

            left_y, right_y = y[left_mask], y[right_mask]
            total_loss = (loss(left_y) * len(left_y) + loss(right_y) * len(right_y)) / n_samples

            if total_loss < best_loss:
                best_loss = total_loss
                best_col = col
                best_split = split

    return best_col, best_split


class DecisionTree621:
    def __init__(self, min_samples_leaf=1, max_features=1.0, loss=None):        
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  
        self.loss = loss 
        
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf:
            return self.create_leaf(y)
        col, split = find_best_split(X, y, self.loss, self.min_samples_leaf, self.max_features)
        
        if col == -1:
            return self.create_leaf(y)
        left_mask = X[:, col] < split
        lchild = self.fit_(X[left_mask], y[left_mask])
        rchild = self.fit_(X[~left_mask], y[~left_mask])
        
        return DecisionNode(col, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        y_results = []
        for x_test in X_test:
            result = self.root.predict(x_test)
            y_results.append(result)

        return np.array(y_results)

         

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=1.0):
        super().__init__(min_samples_leaf=min_samples_leaf, max_features=max_features, loss=np.var)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        predicted_y = self.predict(X_test)
        return r2_score(y_test, predicted_y) 


    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y,np.mean(y))



class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=1.0):
        super().__init__(min_samples_leaf=min_samples_leaf, max_features=max_features, loss=gini)
        self.max_features = max_features

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        predicted_y = self.predict(X_test)
        
        return  accuracy_score(predicted_y,y_test)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        common_class = Counter(y).most_common(1)[0][0]
        return LeafNode(y, common_class)
