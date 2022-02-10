from typing import List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# This code is taken from https://github.com/xinhe97/StepwiseSelectionOLS

# my stepwise selection + sklearn style
class ForwardStepwiseOLS(BaseEstimator, ClassifierMixin):
    def __init__(self, fK: int = 10, add_intercept: bool = True):
        self.fK = fK  # number of predictors
        self.add_intercept = add_intercept
        self.is_fitted_ = False
        self.best_model_fwd = None
        self.best_predictors = None

    @staticmethod
    def my_bic(n, mse, k):
        if k <= 0:
            return np.nan
        else:
            return n * np.log(mse) + k * np.log(n)

    ################### Criterion ###################
    def process_subset(self, X: np.ndarray, y: np.ndarray, feature_index: List[int]):
        # Fit model on feature_set and calculate rsq_adj
        regr = sm.OLS(y, X[:, feature_index]).fit()
        rsq_adj = regr.rsquared_adj
        bic = type(self).my_bic(X.shape[0], regr.mse_resid, len(feature_index))
        rsq = regr.rsquared
        return {"model": regr, "rsq_adj": rsq_adj, "bic": bic, "rsq": rsq, "predictors_index": feature_index}

    ################### Forward Stepwise ###################
    def forward(self, predictors_index: List[int], X: np.ndarray, y: np.ndarray):
        # Pull out predictors we still need to process
        remaining_predictors_index = [p for p in range(X.shape[1])
                                      if p not in predictors_index]

        results = []
        for p in remaining_predictors_index:
            new_predictors_index = predictors_index + [p]
            # new_predictors_index.sort()
            results.append(self.process_subset(X, y, new_predictors_index))
            # Wrap everything up in a nice dataframe
        models = pd.DataFrame(results)
        # Choose the model with the highest rsq_adj
        # best_model = models.loc[models['bic'].idxmin()]
        best_model = models.loc[models['rsq'].idxmax()]
        # Return the best model, along with model's other  information
        return best_model

    def forward_k(self, X_est, y_est, fK):
        models_fwd = pd.DataFrame(columns=["model", "rsq_adj", "bic", "rsq", "predictors_index"])
        predictors_index = []

        m = min(fK, X_est.shape[1])

        for i in range(1, m + 1):
            print(i)
            models_fwd.loc[i] = self.forward(predictors_index, X_est, y_est)
            predictors_index = models_fwd.loc[i, 'predictors_index']

        print(models_fwd)
        # best_model_fwd = models_fwd.loc[models_fwd['bic'].idxmin(),'model']
        best_model_fwd = models_fwd.loc[models_fwd['rsq'].idxmax(), 'model']
        # best_predictors = models_fwd.loc[models_fwd['bic'].idxmin(),'predictors_index']
        best_predictors = models_fwd.loc[models_fwd['rsq'].idxmax(), 'predictors_index']
        return best_model_fwd, best_predictors

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        if self.add_intercept:
            X = sm.add_constant(X, prepend=False)

        # hexin
        self.best_model_fwd, self.best_predictors = self.forward_k(X, y, self.fK)
        # hexin

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        if self.add_intercept:
            X = sm.add_constant(X, prepend=False)

        # hexin
        y_pred = self.best_model_fwd.predict(X[:, self.best_predictors])
        # hexin

        check_is_fitted(self, 'is_fitted_')
        return y_pred

    # These are not needed since the Base class implementation is good.
    # def get_params(self, deep=True):
    #     parent_params = super(ForwardStepwiseOLS, self).get_params(deep)
    #     parent_params["fK"] = self.fK
    #     return parent_params

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self

    def score(self, X, y, sample_weight=None):
        return r2_score(y, self.predict(X), sample_weight=sample_weight)


if __name__ == '__main__':
    ###### DGP ######
    # Spare signals
    N = 1000
    P = 10  # Total number of inputs

    N_true_inputs = 5
    N_false_inputs = P - N_true_inputs
    n_obs = N / 2
    n_pred = N / 2
    error_sd = 1

    # True inputs have coefficient 1
    beta = np.matrix(np.zeros((P, 1)))
    beta[:N_true_inputs, :] = 1

    # Simulate the data
    X = np.matrix(np.random.rand(N, P))
    epsilon = np.matrix(error_sd * np.random.normal(0, size=(N, 1)))
    y = X * beta + epsilon

    # Pack the data into a dataframe
    DF = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    new_names_true = ['x_true_' + str(i) for i in range(1, N_true_inputs + 1)]
    new_names_false = ['x_false_' + str(i) for i in range(1, N_false_inputs + 1)]
    names = new_names_true + new_names_false + ['y']
    DF.columns = names

    # Now we split the data into an estimation and prediction sample. # Randomly draw n_obs observations
    train_index = random.sample(range(0, N), np.int(n_obs))
    train_index.sort()
    DF_estimation = DF.loc[train_index, :]
    DF_prediction = DF.drop(index=train_index)

    ###### Algorithm ######
    fwd = ForwardStepwiseOLS(fK=10)
    fwd.fit(DF_estimation.drop('y', 1), DF_estimation['y'])
    fwd.predict(DF_prediction.drop('y', 1))
    print(fwd.score(DF_prediction.drop('y', 1), DF_prediction['y']))
