import numpy as np
from load_data import load_learnable
from basic_models import validate_models
from feature_importance import perm_feature_importance
from model_evaluation import model_residuals
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer

models = [
    ('LR', LinearRegression(normalize=True)),
    ('BRidge', BayesianRidge(normalize=True)),
    ('DT-5', DecisionTreeRegressor(max_depth=5)),
    ('DT-10', DecisionTreeRegressor(max_depth=10)),
    # ('RF-5', RandomForestRegressor(max_depth=5)),
    # ('RF-10', RandomForestRegressor(max_depth=10)),
    # ('XGB-5', GradientBoostingRegressor(max_depth=5)),
    # ('XGB-10', GradientBoostingRegressor(max_depth=10)),
    # ('Extra', ExtraTreesRegressor()),
]
X, y, features, times = load_learnable()
N = X.shape[0] // len(np.unique(times))
best_mdls = validate_models(X[:-2*N], y[:-2*N], times[:-2*N], models=models)
perm_feature_importance(X[-2*N:], y[-2*N:], features, models=best_mdls)
model_residuals(X, y, times, features, models=best_mdls)
