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
y_pred,sorted_idx,perm_importance_means = re_model.calc_XGBoost(X_train, X_test, y_train, y_test,now)

#############
re_plot.plot_prediction_plus_importance(y_test,y_pred, fn[sorted_idx], perm_importance_means, "XGBoost",now)