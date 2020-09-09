import numpy as np
from load_data import load_learnable
from basic_models import validate_models
from feature_importance import perm_feature_importance
from model_evaluation import model_residuals
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

models = [
    ('LR', LinearRegression(normalize=True)),
    ('BRidge', BayesianRidge(normalize=True)),
    ('DT-5', DecisionTreeRegressor(max_depth=5)),
    ('DT-10', DecisionTreeRegressor(max_depth=10)),
    ('RF-5', RandomForestRegressor(max_depth=5)),
    ('RF-10', RandomForestRegressor(max_depth=10)),
    ('XGB-5', GradientBoostingRegressor(max_depth=5)),
    ('XGB-10', GradientBoostingRegressor(max_depth=10)),
    ('Extra', ExtraTreesRegressor()),
]

# load arrays in the format sklearn expects
X, y, features, times = load_learnable()

# possible addition - transform the features

# split to train and test
N = X.shape[0] // len(np.unique(times))
train = np.ones(X.shape[0]).astype(bool)
train[-2*N:] = False
test = ~train

# validate models and get best performing model from each category
best_mdls = validate_models(X[train], y[train], times[train], models=models)

# plot residuals vs. each attribute for each model from the validation
model_residuals(X, y, times, features, models=best_mdls)

# train the models on the full data and get their permutation feature importance scores
trained_mdls = perm_feature_importance((X[train], y[train]), (X[test], y[test]), features, models=models)
