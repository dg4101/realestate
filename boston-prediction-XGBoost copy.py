from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# datasetのインスタンスを宣言
boston = load_boston()

#boston = load_wine()
#　説明変数と目的変数を生成
X = boston.data
#X = stats.trimboth(X, 0.1)
y = boston.target
#y = stats.trimboth(y, 0.1)
# データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100,test_size=0.20)

print('データセットのレコード数: ', len(X), '\n',
      'トレーニングデータのレコード数: ', len(X_train), '\n',
      'テストデータのレコード数: ', len(X_test))

print("X:" + str(X.shape))
print("X:" + str(X[:2]))
print("X_train:" + str(X_train.shape))
print("X_test:" + str(X_test.shape))
print("y:" + str(y.shape))
print("y:" + str(y[:2]))
print("y_train:" + str(y_train.shape))
print("y_test:" + str(y_test.shape))


#############
from sklearn.preprocessing import MinMaxScaler
# 正規化
mmscaler = MinMaxScaler()

#X_train = mmscaler.fit_transform(X_train)
#X_test = mmscaler.fit_transform(X_test)


#############
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

#param_grid = {'min_samples_leaf': [1, 3, 5, 7, 10]}
param_grid = {
        'eta': [0.01],             # default = 0.3      
        'gamma': [1,2,3],            # default = 0
        'max_depth': [7,8,9],      # default = 6
        'min_child_weight': [1],   # default = 1
        'subsample': [0.8,1.0],        # default = 1
        'colsample_bytree': [0.8,1.0], # default = 1
        }
kf = KFold(n_splits=5, shuffle = True, random_state = 1)

forest = xgb.XGBRegressor(objective ='reg:squarederror')

grid_search = GridSearchCV(forest, param_grid,  cv=kf.split(X_train,y_train), verbose=2, scoring='neg_mean_squared_error',return_train_score=True,n_jobs=2,)

# GridSearchCVは最良パラメータの探索だけでなく、それを使った学習メソッドも持っています
grid_search.fit(X_train, y_train)
best_forest = grid_search.best_estimator_

print('best score by XGBoost: {:0.10f}'.format(grid_search.score(X_test, y_test)))
print('best params: ' + str(grid_search.best_params_))
params = np.array(list(grid_search.best_params_.items()))
print(str(params[0,0]) + ': ' + str(params[0,1]))
print(str(params[1,0]) + ': ' + str(params[1,1]))
print('best validated score:  {:0.10f}'.format(grid_search.best_score_))

print('best score by best_estimator_: {:0.10f}'.format(best_forest.score(X_test, y_test)))
#print('best score by oob_score_: {:0.10f}'.format(best_forest.)) # oob_score on RandomForestRegressor needs to be true. default false
# Get price prediction
y_pred = best_forest.predict(X_test)
#print('prediced price:  ' + str(y_pred))
r2 = r2_score(y_test,y_pred)
print('best prediction with best_forest by r2_score:  ' + str(r2))

'''
# Get price prediction
y_pred = grid_search.predict(X_test)
#print('prediced price:  ' + str(y_pred))
r2 = r2_score(y_test,y_pred)
print('best prediction with grid_search by r2_score:  ' + str(r2))
'''

plt.subplot(122)
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(best_forest, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(boston.feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.tight_layout()

import datetime
now = datetime.datetime.now()
plt.savefig('images/boston-' + now.strftime("%Y%m%d-%H%M") + '-XGBoost-.png')
plt.show()
'''


plt.subplot(132)
sorted_idx = best_forest.feature_importances_.argsort()
plt.barh(boston.feature_names[sorted_idx], best_forest.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.tight_layout()

# Re cal predcition with determined params
# Regression with best param
forest = RandomForestRegressor(max_depth=int(params[0,1]),min_samples_leaf=int(params[1,1]))
#forest = RandomForestRegressor(min_samples_leaf=int(params[0,1]))
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
#print('prediced price:  ' + str(y_pred))
r2 = r2_score(y_test,y_pred)
print('Re-cal best prediction by r2_score:  ' + str(r2))

# Regression with default
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
#print('prediced price:  ' + str(y_pred))
r2 = r2_score(y_test,y_pred)
print('best prediction without GS,CV by r2_score:  ' + str(r2))
'''

'''
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7],
              'min_samples_leaf': [1, 3, 5, 7, 10]}



from sklearn.ensemble import RandomForestClassifier







##############################
# Convert to 2 dimention
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X2D = pca.fit_transform(boston.data)
print(X2D.shape)
print(X2D.reshape(-1, 1).shape)


###############################
from sklearn import linear_model #線形回帰モジュールのインポート
pca_data = X2D.reshape(-1, 1)
model = linear_model.LinearRegression() #回帰モデル
model.fit(pca_data, y) #データを入れて訓練

#テストデータの作成
print("Print train min and max")
print(pca_data.min())
print(pca_data.max())
print(np.arange(pca_data.min(), pca_data.max(), 0.1))

from pandas import DataFrame #pandasモジュール
properties=DataFrame(np.arange(pca_data.min(), pca_data.max(), 0.1))
print("Print price test")
print(model.predict(properties))
prices_predict=model.predict(properties)

plt.scatter(np.array(pca_data), y, alpha=0.5) #訓練データの散布図
plt.plot(properties.values, prices_predict, c="r") #回帰直線
plt.title("Boston House Prices dataset")
plt.xlabel("PCA data")
plt.ylabel("prices")
plt.show()
'''