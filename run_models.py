import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


import seaborn as sns

def predict(model):
    train = pd.read_csv("train.csv", low_memory=False)
    store = pd.read_csv("store.csv")
    test = pd.read_csv("test.csv")

    store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
    store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])
    store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])

    store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
    store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])
    store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])

    train['Date'] = pd.to_datetime(train['Date'])

    train['Year'] = train['Date'].dt.year
    train['Month'] = train['Date'].dt.month
    train['Day'] = train['Date'].dt.day
    # train['WeekOfYear'] = train['Date'].dt.isocalendar().week

    train['SalesPerCustomers'] = train['Sales'] / train['Customers']

    train = train[(train["Open"] != 0) & (train['Sales'] != 0)]
    train.dropna()
    train.isnull().sum()
    train.dropna()
    test.isnull().sum()
    pd.isna(store)
    store.dropna()

    train_store = train.merge(store, on='Store', how='left')

    train_store.dropna()

    cat_cols = train_store.select_dtypes(include=['object']).columns

    for i in cat_cols:
        print(i)
        print(train_store[i].value_counts())
        print('-' * 20)

    train_store['StateHoliday'] = train_store['StateHoliday'].map({'0': 0, 0: 0, 'a': 1, 'b': 2, 'c': 3})
    train_store['StateHoliday'] = train_store['StateHoliday'].astype('int', errors='ignore')
    train_store['StoreType'] = train_store['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
    train_store['StoreType'] = train_store['StoreType'].astype(int)
    train_store['Assortment'] = train_store['Assortment'].map({'a': 1, 'b': 2, 'c': 3})
    train_store['Assortment'] = train_store['Assortment'].astype(int)
    train_store['PromoInterval'] = train_store['PromoInterval'].map({'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3})
    train_store['PromoInterval'] = train_store['PromoInterval'].astype(int)

    X = train_store.drop(['Sales', 'Date', 'Customers', 'SalesPerCustomers'], 1)
    # Transform Target Variable
    y = np.log(train_store['Sales'] + 1)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=1)

    X_train.shape, X_val.shape, y_train.shape, y_val.shape

    if(model == 1):

        dt = DecisionTreeRegressor(max_depth=11)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_val)

        y_pred_dt = np.exp(y_pred_dt) - 1
        y_val = np.exp(y_val) - 1
        from sklearn.metrics import r2_score, mean_squared_error

        print(r2_score(y_val, y_pred_dt))
        print(np.sqrt(mean_squared_error(y_val, y_pred_dt)))

        def ToWeight(y):
            w = np.zeros(y.shape, dtype=float)
            ind = y != 0
            w[ind] = 1. / (y[ind] ** 2)
            return w

        def rmspe(y, yhat):
            w = ToWeight(y)
            rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
            return rmspe

        rmspe(y_val, y_pred_dt)

        def get_rmspe_score(model, input_values, y_actual):
            y_predicted = model.predict(input_values)
            y_actual = np.exp(y_actual) - 1
            y_predicted = np.exp(y_predicted) - 1
            score = rmspe(y_actual, y_predicted)
            return score

        from sklearn.model_selection import RandomizedSearchCV

        params = {
            'max_depth': list(range(5, 25))
        }

        base = DecisionTreeRegressor()

        model_tuned = RandomizedSearchCV(base, params, return_train_score=True).fit(X_train, y_train)

        model_cv_results = pd.DataFrame(model_tuned.cv_results_).sort_values(by='mean_test_score', ascending=False)
        model_cv_results

        import matplotlib.pyplot as plt

        model_cv_results.set_index('param_max_depth')['mean_test_score'].plot(color='g', legend=True)
        model_cv_results.set_index('param_max_depth')['mean_train_score'].plot(color='r', legend=True)
        plt.grid(True)

        model_alg = dt


    elif(model==2):
        from sklearn.ensemble import AdaBoostRegressor
        adaboost_tree = AdaBoostRegressor(DecisionTreeRegressor())
        adaboost_tree.fit(X_train, y_train)

        y_hat = adaboost_tree.predict(X_val)

        model_alg = adaboost_tree

    elif(model==3):

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalidate = xgb.DMatrix(X_val[X_train.columns], y_val)

        params = {
            'eta': 1,
            'max_depth': 5,
            'objecive': 'reg:linear'
        }

        model_xg = xgb.train(params, dtrain, 5)

        y_pred_xg = model_xg.predict(dvalidate)

        y_pred_xg = np.exp(y_pred_xg) - 1

        xgbr = xgb.XGBRegressor(verbosity=0)
        print(xgbr)
        xgbr.fit(X_train, y_train)
        model_alg =xgbr
    elif (model == 4):
        randomForest = RandomForestRegressor(n_estimators=25, n_jobs=-1, verbose=1)
        randomForest.fit(X_train, y_train)
        model_alg = randomForest

    test_cust = train.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)
    print(test_cust)

    test_1 = test.merge(test_cust, on='Store', how='left')
    print(test_1.head())

    test_m = test_1.merge(store, on='Store', how='left')
    test_m.shape
    print(test_m)

    test_m['Open'].fillna(1, inplace=True)

    test_m['Date'] = pd.to_datetime(test_m['Date'])

    test_m['Day'] = test_m['Date'].dt.day
    test_m['Month'] = test_m['Date'].dt.month
    test_m['Year'] = test_m['Date'].dt.year

    test_m.drop('Date', 1, inplace=True)

    cat_cols = test_m.select_dtypes(include=['object']).columns

    for i in cat_cols:
        print(i)
        print(test_m[i].value_counts())
        print('-' * 20)

    test_m['StateHoliday'] = test_m['StateHoliday'].map({'0': 0, 'a': 1})
    test_m['StateHoliday'] = test_m['StateHoliday'].astype(int)

    test_m['StoreType'] = test_m['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
    test_m['StoreType'] = test_m['StoreType'].astype(int)

    test_m['Assortment'] = test_m['Assortment'].map({'a': 1, 'b': 2, 'c': 3})
    test_m['Assortment'] = test_m['Assortment'].astype(int)

    test_m['PromoInterval'] = test_m['PromoInterval'].map(
        {'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3})
    test_m['PromoInterval'] = test_m['PromoInterval'].astype(int)

    test_pred = model_alg.predict(test_m[X_train.columns])
    test_pred_inv = np.exp(test_pred) - 1
    print(test_pred_inv)

    submission = pd.DataFrame({'Id': test_m['Id'], 'Sales': test_pred_inv})
    submission['Sales'] = submission['Sales'].astype(int)
    submission['Id'] = submission.index
    submission['Id'] = submission['Id'] + 1
    print(submission.head())
    test_m['Sales'] = test_pred_inv
    test_m['Sales'] = test_m['Sales'].astype(int)

    return test_m