import re_load_data
import re_model
import re_plot
import datetime

trim = 0.1 # trim data
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

##########
# Run RandomForestRegressor
y_pred,sorted_idx,perm_importance_means = re_model.calc_RandomForestRegressor(X_train, X_test, y_train, y_test,now)

#############
re_plot.plot_prediction_plus_importance(y_test,y_pred, fn[sorted_idx], perm_importance_means, "RandomForestRegressor",now)

#############################
'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
'''

'''
fig = plt.figure(dpi=400)
fig.suptitle("Boston House Prices Prediction by RandomForestRegressor")
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
plt.barh(fn[sorted_idx], perm_importance_means)
plt.xlabel("Permutation Importance")
plt.tight_layout()

import datetime
now = datetime.datetime.now()
plt.savefig('images/boston-' + now.strftime("%Y%m%d-%H%M") + '-RandomForestRegressor-.png')
plt.show()

''''''


plt.subplot(132)
sorted_idx = best_forest.feature_importances_.argsort()
plt.barh(boston.feature_names[sorted_idx], best_forest.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.tight_layout()

# Re cal predcition with determined params
# Regression with best param
forest = RandomForestRegressor(max_depth=int(params[0,1]),min_samples_leaf=int(params[1,1]))
#forest = RandomForestRegressor(min_samples_leaf=int(params[0,1]))
forest.fit(X_train_scaled, y_train)
y_pred = forest.predict(X_test)
#print('prediced price:  ' + str(y_pred))
r2 = r2_score(y_test,y_pred)
print('Re-cal best prediction by r2_score:  ' + str(r2))

# Regression with default
forest = RandomForestRegressor()
forest.fit(X_train_scaled, y_train)
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