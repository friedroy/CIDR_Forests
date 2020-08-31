from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from load_data import load_csv_tensor, tensor_to_features
import numpy as np
import matplotlib.pyplot as plt
import shap


tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope'], return_years=True)
X, y, features = tensor_to_features(tens, f2i, lookback=1, remove_att=True)

n_years = 30
N = tens.shape[0]*n_years
X_val, y_val = X[N:N+tens.shape[0]], y[N:N+tens.shape[0]]
X, y = X[:N], y[:N]

model = LinearRegression(normalize=True).fit(X, y)
score = model.score(X_val, y_val)
print(score)

res = permutation_importance(model, X_val, y_val, n_repeats=30)
means = []
stds = []
for i in range(len(features)):
    means.append(max(res.importances_mean[i], 0))
    stds.append(res.importances_std[i] if res.importances_mean[i] > 0 else 0)
inds = np.argsort(means)
means, stds = np.array(means)[inds], np.array(stds)[inds]
names = np.array(features)[inds]

plt.figure()
plt.barh(np.arange(len(features)), means, xerr=stds, capsize=5, tick_label=features)
# plt.xticks(np.arange(len(features)), names, rotation=45)
plt.xlabel('perm. feature importance')
plt.show()

# explainer = shap.TreeExplainer(model)
# shap_vals = explainer.shap_values(X_val[:50], y_val[:50])
# shap.summary_plot(shap_vals, X_val[:50], plot_type="bar", feature_names=features)
# shap.force_plot(explainer.expected_value, shap_vals, X_val[:50], feature_names=features)
# plt.figure(dpi=300)
# shap.summary_plot(shap_vals, X_val[:50], feature_names=features)
# plt.show()
