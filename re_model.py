from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error

def calc_RandomForestRegressor(X_train, X_test, y_train, y_test,now):
    from sklearn.ensemble import RandomForestRegressor

    #param_grid = {'min_samples_leaf': [1, 3, 5, 7, 10]}
    param_grid = {'max_depth': [1,2, 3, 4, 5, 6, 7],
                'min_samples_leaf': [1, 3, 5, 7, 10]}
    forest = RandomForestRegressor()
    kf = KFold(n_splits=5, shuffle = True, random_state = 1)
    #grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="r2",return_train_score=True)
    grid_search = GridSearchCV(forest, param_grid, cv=kf, verbose=1, scoring="r2",return_train_score=True)
    grid_search.fit(X_train, y_train)

    '''
    # Nested cv
    from sklearn.model_selection import cross_val_score
    ns_grid_search = GridSearchCV(forest, param_grid, cv=kf, verbose=1, scoring="r2",return_train_score=True)
    ns_scores = cross_val_score(ns_grid_search, X=X, y=y, cv=kf)
    print('best validated nested cv score:  {:0.10f}'.format(ns_scores.mean()))
    '''

    # Fit the model
    y_pred, sorted_idx, perm_importance_means = fit_model(grid_search,X_test, y_test)

    # Display result
    display_result(grid_search,X_test, y_test,y_pred,"RandomForestRegressor",now)

    return y_pred, sorted_idx, perm_importance_means

def calc_SVR(X_train, X_test, y_train, y_test,now):
    from sklearn.svm import SVR
    import numpy as np

    forest = SVR()
    kf = KFold(n_splits=5, shuffle = True, random_state = 1)
    params_cnt = 20
    param_grid = {"C":np.logspace(0,2,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}

    grid_search = GridSearchCV(forest, param_grid, cv=kf, verbose=1, scoring="r2",return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Fit the model
    y_pred, sorted_idx, perm_importance_means = fit_model(grid_search,X_test, y_test)

    # Display result
    display_result(grid_search,X_test, y_test,y_pred,"Epsilon-Support Vector Regression",now)

    return y_pred, sorted_idx, perm_importance_means

def calc_LightGBM(X_train, X_test, y_train, y_test,now):
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import mean_absolute_error,mean_squared_error

    forest = lgb.LGBMRegressor()
    kf = KFold(n_splits=5, shuffle = True, random_state = 1)
    param_grid = {"max_depth": [10, 25, 50, 75],
                    "learning_rate" : [0.001,0.01,0.05,0.1],
                    "num_leaves": [100,300,900,1200],
                    "n_estimators": [100,200,500]
                    }
    
    grid_search = GridSearchCV(forest, param_grid, cv=kf, verbose=1, scoring="neg_mean_squared_error",return_train_score=True)
    '''
    grid_search.fit(X_train, y_train)
    '''
    grid_search.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=30)
    # Fit the model
    y_pred, sorted_idx, perm_importance_means = fit_model(grid_search,X_test, y_test)

    # Display result
    display_result(grid_search,X_test, y_test,y_pred,"LightGBM",now)

    return y_pred, sorted_idx, perm_importance_means

def calc_XGBoost(X_train, X_test, y_train, y_test,now):
    import xgboost as xgb
    import numpy as np

    forest = xgb.XGBRegressor()

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
    
    #grid_search = GridSearchCV(forest, param_grid, cv=kf, verbose=1, scoring="neg_mean_squared_error",return_train_score=True)
    grid_search = GridSearchCV(forest, param_grid, cv=kf, verbose=1, scoring="r2",return_train_score=True)
    #grid_search.fit(X_train, y_train)
    grid_search.fit(X_train, y_train,
        eval_metric='rmse')
    # Fit the model
    y_pred, sorted_idx, perm_importance_means = fit_model(grid_search,X_test, y_test)

    # Display result
    display_result(grid_search,X_test, y_test,y_pred,"XGBoost",now)

    return y_pred, sorted_idx, perm_importance_means

def fit_model(grid_search,X_test, y_test):

    # Get best estimaor
    best_forest = grid_search.best_estimator_
    # Get price prediction
    y_pred = best_forest.predict(X_test)

    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(best_forest, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    perm_importance_means = perm_importance.importances_mean[sorted_idx]

    return y_pred, sorted_idx, perm_importance_means

############################
def display_result(grid_search,X_test, y_test,y_pred,algorism,now):

    r2 = r2_score(y_test,y_pred)
    # Show best score and params
    with open('text/boston-' + now + '-' + algorism + '.txt', 'w') as f:
        print(algorism + ' best score: {:0.10f}'.format(grid_search.score(X_test, y_test)), file=f)
        print('best params: ' + str(grid_search.best_params_), file=f)
        print('best validated score:  {:0.10f}'.format(grid_search.best_score_), file=f)

        #print('prediced price:  ' + str(y_pred))
        print('best prediction with best_forest by r2_score:  ' + str(r2), file=f)
        # Display MAE RMSE score
        print('mean_absolute_error MAE:',mean_absolute_error(y_test,y_pred), file=f)
        print('mean_squared_error RMSE:',mean_squared_error(y_test,y_pred), file=f)

    '''
    # Get price prediction
    y_pred = grid_search.predict(X_test)
    #print('prediced price:  ' + str(y_pred))
    r2 = r2_score(y_test,y_pred)
    print('best prediction with grid_search by r2_score:  ' + str(r2))
    '''    
