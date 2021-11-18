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



