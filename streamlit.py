import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import streamlit as st


def main():
    menu = ["Introduction", "Data Exploration", "Models", "Conclusion"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Introduction":
        st.write("""# Data Science Assignment""")
        st.write("""# Introduction""")
        st.subheader("Files")
        st.write("""
        train.csv - historical data including Sales
        
        test.csv - historical data excluding Sales
        
        store.csv - supplemental information about the stores""")
        st.subheader("Data fields")
        st.write("""
                Id - an Id that represents a (Store, Date) duple within the test set
                
                Store - a unique Id for each store
                
                Sales - the turnover for any given day (this is what you are predicting)
                
                Customers - the number of customers on a given day
                
                Open - an indicator for whether the store was open: 0 = closed, 1 = open
                
                StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
                
                SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
                
                StoreType - differentiates between 4 different store models: a, b, c, d
                
                Assortment - describes an assortment level: a = basic, b = extra, c = extended
                
                CompetitionDistance - distance in meters to the nearest competitor store
                
                CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
                
                Promo - indicates whether a store is running a promo on that day
                
                Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
                
                Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
                
                PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that stor
    """)

    elif choice == "Data Exploration":
        st.write("""# Data Exploration""")

        st.write("Our Imports: ")
        st.code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import streamlit as st""")

        train = pd.read_csv("train.csv", low_memory=False)
        st.write("Train: ", train.head())
        st.code("train = pd.read_csv('train.csv', low_memory=False)")

        store = pd.read_csv("store.csv")
        st.write("Store: ", store.head())
        st.code("store = pd.read_csv('store.csv')")

        test = pd.read_csv("test.csv")
        st.write("Test: ", test.head())
        st.code("test = pd.read_csv('test.csv')")

        # train['Date'] = pd.to_datetime(train['Date'])
        # train['Year'] = train['Date'].dt.year
        # train['Month'] = train['Date'].dt.month
        # train['Day'] = train['Date'].dt.day
        # # train['WeekOfYear'] = train['Date'].dt.isocalendar().week
        #
        # train['SalesPerCustomers'] = train['Sales'] / train['Customers']
        #
        # print(train['SalesPerCustomers'].describe())
        # print(store.isnull().sum())
        # # check if we have negative Sales
        # sales_minus = train[(train["Sales"] < 0)]
        # print(sales_minus)
        #
        # train = train[(train["Open"] != 0) & (train['Sales'] != 0)]
        #
        # store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
        # store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])
        # store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])
        #
        # store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
        # store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(
        # store['CompetitionOpenSinceMonth'].mode().iloc[0])
        # store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(
        # store['CompetitionOpenSinceYear'].mode().iloc[0])
        #
        # train.isnull().sum()
        # train.dropna()
        # train.isnull().sum()
        # train.dropna()
        # test.isnull().sum()
        # pd.isna(store)
        # store.dropna()
        #
        # train_store = train.merge(store, on='Store', how='left')
        #
        # train_store.dropna()
        #
        # sns.catplot(data=train_store, x='Month', y="Sales",
        #             col='StoreType',  # per store type in cols
        #             palette='plasma',
        #             hue='StoreType',
        #             kind='bar',
        #             row='Promo')
        #
        # sns.catplot(data=train_store, x='Month', y="Customers",
        #             col='StoreType',  # per store type in cols
        #             palette='plasma',
        #             hue='StoreType',
        #             kind='bar',
        #             row='Promo')
        #
        # # StoreType pie
        # store_types = store['StoreType'].value_counts().sort_values(ascending=False)
        # ax = store_types.plot.pie(autopct="%.1f%%", startangle=90, figsize=(10, 10))
        # ax.set_title('StoreType pie chart')
        #
        # # DayOfWeek vs Sales
        # fig, ax = plt.subplots(figsize=(15, 10))
        # sns.barplot(x="DayOfWeek", y="Sales", data=train)
        #
        # cat_cols = train_store.select_dtypes(include=['object']).columns
        #
        # for i in cat_cols:
        #     print(i)
        #     print(train_store[i].value_counts())
        #     print('-' * 20)
        #
        # train_store['StateHoliday'] = train_store['StateHoliday'].map({'0': 0, 0: 0, 'a': 1, 'b': 2, 'c': 3})
        # train_store['StateHoliday'] = train_store['StateHoliday'].astype('int', errors='ignore')
        # train_store['StoreType'] = train_store['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
        # train_store['StoreType'] = train_store['StoreType'].astype(int)
        # train_store['Assortment'] = train_store['Assortment'].map({'a': 1, 'b': 2, 'c': 3})
        # train_store['Assortment'] = train_store['Assortment'].astype(int)
        # train_store['PromoInterval'] = train_store['PromoInterval'].map({'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3})
        # train_store['PromoInterval'] = train_store['PromoInterval'].astype(int)
        #
        # train_store.dtypes



    elif choice == "Models":
        st.write("""# Models""")
    elif choice == "Conclusion":
        st.write("""# Conclusion""")


if __name__ == '__main__':
    main()
