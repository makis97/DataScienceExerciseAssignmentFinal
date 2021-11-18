import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from IPython.display import display
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
# reading data
import seaborn
from sklearn.ensemble import AdaBoostRegressor

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
#train['WeekOfYear'] = train['Date'].dt.isocalendar().week

train['SalesPerCustomers'] = train['Sales'] / train['Customers']

print(train['SalesPerCustomers'].describe())
print(store.isnull().sum())
# check if we have negative Sales
sales_minus = train[(train["Sales"] < 0)]
print(sales_minus)

train = train[(train["Open"] != 0) & (train['Sales'] != 0)]

train.isnull().sum()
train.dropna()
train.isnull().sum()
train.dropna()
test.isnull().sum()
pd.isna(store)
store. dropna()

train_store = train.merge(store , on='Store' , how='left')

train_store.dropna()

sns.catplot(data = train_store, x = 'Month', y = "Sales",
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               kind = 'bar',
               row = 'Promo')

sns.catplot(data = train_store, x = 'Month', y = "Customers",
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               kind = 'bar',
               row = 'Promo')

#StoreType pie
store_types = store['StoreType'].value_counts().sort_values(ascending=False)
ax= store_types.plot.pie(autopct="%.1f%%",startangle=90, figsize=(10,10))
ax.set_title('StoreType pie chart')

# DayOfWeek vs Sales
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(x="DayOfWeek", y="Sales", data=train)

cat_cols = train_store.select_dtypes(include=['object']).columns

for i in cat_cols:
    print(i)
    print(train_store[i].value_counts())
    print('-'*20)

train_store['StateHoliday'] = train_store['StateHoliday'].map({'0':0 , 0:0 , 'a':1 , 'b':2 , 'c':3})
train_store['StateHoliday'] = train_store['StateHoliday'].astype('int', errors='ignore')
train_store['StoreType'] = train_store['StoreType'].map({'a':1 , 'b':2 , 'c':3 , 'd':4})
train_store['StoreType'] = train_store['StoreType'].astype(int)
train_store['Assortment'] = train_store['Assortment'].map({'a':1 , 'b':2 , 'c':3})
train_store['Assortment'] = train_store['Assortment'].astype(int)
train_store['PromoInterval'] = train_store['PromoInterval'].map({'Jan,Apr,Jul,Oct':1 , 'Feb,May,Aug,Nov':2 , 'Mar,Jun,Sept,Dec':3})
train_store['PromoInterval'] = train_store['PromoInterval'].astype(int)

train_store.dtypes

X = train_store.drop(['Sales','Date','Customers','SalesPerCustomers'],1)
#Transform Target Variable
y = np.log(train_store['Sales']+1)

from sklearn.model_selection import train_test_split
X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.30 , random_state = 1 )

X_train.shape , X_val.shape , y_train.shape , y_val.shape

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=11)
dt.fit(X_train , y_train)
y_pred_dt = dt.predict(X_val)

y_pred_dt = np.exp(y_pred_dt)-1
y_val = np.exp(y_val)-1
from sklearn.metrics import r2_score , mean_squared_error

print(r2_score(y_val , y_pred_dt))
print(np.sqrt(mean_squared_error(y_val , y_pred_dt)))

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

rmspe(y_val,y_pred_dt)

def get_rmspe_score(model, input_values, y_actual):
    y_predicted=model.predict(input_values)
    y_actual=np.exp(y_actual)-1
    y_predicted=np.exp(y_predicted)-1
    score=rmspe(y_actual, y_predicted)
    return score

from sklearn.model_selection import RandomizedSearchCV

params = {
    'max_depth' : list(range(5,25))
}

base  = DecisionTreeRegressor()

model_tuned = RandomizedSearchCV(base , params , return_train_score=True).fit(X_train , y_train)

model_cv_results = pd.DataFrame(model_tuned.cv_results_).sort_values(by='mean_test_score' , ascending=False)
model_cv_results

import matplotlib.pyplot as plt

model_cv_results.set_index('param_max_depth')['mean_test_score'].plot(color='g',legend=True)
model_cv_results.set_index('param_max_depth')['mean_train_score'].plot(color='r' , legend=True)
plt.grid(True)

test_cust = train.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)
print(test_cust)

test_1 = test.merge(test_cust , on='Store' , how='left')
print(test_1.head())

test_m = test_1.merge(store , on='Store' , how='left')
test_m.shape
print(test_m)

test_m['Open'].fillna(1,inplace=True)

test_m['Date'] = pd.to_datetime(test_m['Date'])

test_m['Day'] = test_m['Date'].dt.day
test_m['Month'] = test_m['Date'].dt.month
test_m['Year'] = test_m['Date'].dt.year

test_m.drop('Date',1,inplace=True)

cat_cols = test_m.select_dtypes(include=['object']).columns

for i in cat_cols:
    print(i)
    print(test_m[i].value_counts())
    print('-'*20)

test_m['StateHoliday'] = test_m['StateHoliday'].map({'0':0 , 'a':1})
test_m['StateHoliday'] = test_m['StateHoliday'].astype(int)

test_m['StoreType'] = test_m['StoreType'].map({'a':1 , 'b':2 , 'c':3 , 'd':4})
test_m['StoreType'] = test_m['StoreType'].astype(int)

test_m['Assortment'] = test_m['Assortment'].map({'a':1 , 'b':2 , 'c':3})
test_m['Assortment'] = test_m['Assortment'].astype(int)

test_m['PromoInterval'] = test_m['PromoInterval'].map({'Jan,Apr,Jul,Oct':1 , 'Feb,May,Aug,Nov':2 , 'Mar,Jun,Sept,Dec':3})
test_m['PromoInterval'] = test_m['PromoInterval'].astype(int)

test_pred = dt.predict(test_m[X_train.columns])
test_pred_inv = np.exp(test_pred)-1
print(test_pred_inv)

submission = pd.DataFrame({'Id' : test_m['Id'] , 'Sales' : test_pred_inv})
submission['Sales'] = submission['Sales'].astype(int)
submission['Id']= submission.index
submission['Id'] = submission['Id']+1
print(submission.head())

from sklearn.ensemble import AdaBoostRegressor
adaboost_tree = AdaBoostRegressor(DecisionTreeRegressor())
adaboost_tree.fit(X_train, y_train)

y_hat = adaboost_tree.predict(X_val)

test_pred2 = adaboost_tree.predict(test_m[X_train.columns])

test_pred_inv2 = np.exp(test_pred2)-1
print(test_pred_inv2)


def plot_importance(model):
    k = list(zip(X, model.feature_importances_))
    k.sort(key=lambda tup: tup[1])

    labels, vals = zip(*k)

    plt.barh(np.arange(len(X)), vals, align='center')
    plt.yticks(np.arange(len(X)), labels)


importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': adaboost_tree.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

sns.barplot(data=importance_df.head(10), x='importance', y='feature')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')


import xgboost as xgb
dtrain = xgb.DMatrix(X_train,y_train)
dvalidate = xgb.DMatrix(X_val[X_train.columns],y_val)

params = {
    'eta' : 1,
    'max_depth' : 5,
    'objecive' : 'reg:linear'
}


model_xg = xgb.train(params, dtrain , 5)

y_pred_xg = model_xg.predict(dvalidate)

y_pred_xg = np.exp(y_pred_xg)-1


rmspe(y_val , y_pred_xg)



xgbr = xgb.XGBRegressor(verbosity=0)
print(xgbr)
xgbr.fit(X_train, y_train)


y_hat4 = xgbr.predict(X_val)

test_pred3 = xgbr.predict(test_m[X_train.columns])

test_pred_inv3 = np.exp(test_pred3)-1
print(test_pred_inv3)


from sklearn.ensemble import RandomForestRegressor
randomForest = RandomForestRegressor(n_estimators=25, n_jobs=-1, verbose=1)
randomForest.fit(X_train, y_train)

y_hat23 = randomForest.predict(X_val)
test_pred23 = randomForest.predict(test_m[X_train.columns])

test_pred_inv32 = np.exp(test_pred23)-1
print(test_pred_inv32)

get_rmspe_score(randomForest, test_m[X_train.columns],test_pred23)

test_m['Sales'] = test_pred_inv32
print(test_m)

sales = train[train.Store == 1].loc[:, ['Date', 'Sales']]

# reverse to the order: from 2013 to 2015
sales = sales.sort_index(ascending = False)

# to datetime64
sales['Date'] = pd.DatetimeIndex(sales['Date'])
print(sales.dtypes)

print(sales.head())

ax = sales.set_index('Date').plot(figsize = (12, 4), color = 'c')
ax.set_ylabel('Daily Number of Sales')
ax.set_xlabel('Date')

test_m['Date'] = test['Date']
print(test_m)

pred_sales = test_m[test_m.Store == 1].loc[:, ['Date', 'Sales']]


pred_sales = pred_sales.sort_index(ascending = False)

# to datetime64
pred_sales['Date'] = pd.DatetimeIndex(pred_sales['Date'])
#pred_sales.dtypes

ax = pred_sales.set_index('Date').plot(figsize = (12, 4), color = 'c')
ax.set_ylabel('Daily Number of Sales')
ax.set_xlabel('Date')

ax2 = sales.set_index('Date').plot(figsize = (12, 4), color = 'c', xlim=['2014-8-1','2014-9-30'])
ax2.set_ylabel('Daily Number of Sales')
ax2.set_xlabel('Date')

ax2 = sales.set_index('Date').plot(figsize = (12, 4), color = 'c', xlim=['2013-8-1','2013-9-30'])
ax2.set_ylabel('Daily Number of Sales')
ax2.set_xlabel('Date')

plt.show()

