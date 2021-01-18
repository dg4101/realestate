import re_load_data
import re_model
import re_plot
import datetime

trim = 0.05 # trim data
now = datetime.datetime.now()
now = now.strftime("%Y%m%d-%H%M-"+ str(trim)) 

# Load data
X_train, X_test, y_train, y_test, fn = re_load_data.bs_load_data(trim)

#############
from sklearn.preprocessing import MinMaxScaler
# 正規化
mmscaler = MinMaxScaler()

X_train = mmscaler.fit_transform(X_train)
X_test = mmscaler.fit_transform(X_test)

#################
# Run Epsilon-Support Vector Regression
y_pred,sorted_idx,perm_importance_means = re_model.calc_SVR(X_train, X_test, y_train, y_test,now)

#############
re_plot.plot_prediction_plus_importance(y_test,y_pred, fn[sorted_idx], perm_importance_means, "SVR",now)

######################################
'''

#############
from sklearn.preprocessing import MinMaxScaler
# 正規化
mmscaler = MinMaxScaler()

X_train = mmscaler.fit_transform(X_train)
X_test = mmscaler.fit_transform(X_test)
#############
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score

forest = SVR()
params_cnt = 20
param_grid = {"C":np.logspace(0,2,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="r2",return_train_score=True)

# GridSearchCVは最良パラメータの探索だけでなく、それを使った学習メソッドも持っています
grid_search.fit(X_train, y_train)
best_forest = grid_search.best_estimator_

print('best score by SVR: {:0.10f}'.format(grid_search.score(X_test, y_test)))
print('best params: ' + str(grid_search.best_params_))
params = np.array(list(grid_search.best_params_.items()))
print(str(params[0,0]) + ': ' + str(params[0,1]))
print(str(params[1,0]) + ': ' + str(params[1,1]))
print('best validated score:  {:0.10f}'.format(grid_search.best_score_))

print('best score by best_estimator_: {:0.10f}'.format(best_forest.score(X_test, y_test)))
# Get price prediction
y_pred = best_forest.predict(X_test)
#print('prediced price:  ' + str(y_pred))
r2 = r2_score(y_test,y_pred)
print('best prediction with best_forest by r2_score:  ' + str(r2))

fig = plt.figure(dpi=400)
fig.suptitle("Boston House Prices Prediction by Epsilon-Support Vector Regression")
plt.get_current_fig_manager().full_screen_toggle()
plt.rcParams["font.size"] = 6

plt.subplot(121)
plt.rcParams["figure.figsize"] = (6, 6)
plt.scatter(y_pred, y_test, alpha=0.5) #訓練データの散布図
plt.plot(y_pred, y_pred, c="r") #回帰直線
plt.xlabel("Predicted Price")
plt.ylabel("Actual prices")
plt.xlim(5,50)
plt.ylim(5,50)
plt.tight_layout()

plt.subplot(122)
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(best_forest, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(boston.feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.tight_layout()

import datetime
now = datetime.datetime.now()
plt.savefig('images/boston-' + now.strftime("%Y%m%d-%H%M") + '-SVR-.png')
plt.show()
'''