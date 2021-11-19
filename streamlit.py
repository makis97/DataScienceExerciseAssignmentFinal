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

        st.write("##### Imports ")
        st.code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import streamlit as st""")

        train = pd.read_csv("train.csv", low_memory=False)
        st.write("##### Reading Train CSV File: ", train.head())
        st.code("train = pd.read_csv('train.csv', low_memory=False)")

        store = pd.read_csv("store.csv")
        st.write("##### Reading Store CSV File: ", store.head())
        st.code("store = pd.read_csv('store.csv')")

        test = pd.read_csv("test.csv")
        st.write("##### Reading Test CSV File: ", test.head())
        st.code("test = pd.read_csv('test.csv')")

        st.write("##### Reading and Splitting Date into Year, Month, Day Columns in Train CSV File. ")
        st.code("""train['Date'] = pd.to_datetime(train['Date'])
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day""")
        train['Date'] = pd.to_datetime(train['Date'])
        train['Year'] = train['Date'].dt.year
        train['Month'] = train['Date'].dt.month
        train['Day'] = train['Date'].dt.day
        st.write("Result after splitting Date: ", train.head())

        st.write("##### Calculate the average of Sales Per Customers in Train CSV File. On average customers spend about 9.50$ per day.")
        train['SalesPerCustomers'] = train['Sales'] / train['Customers']
        st.write("Average of Sales Per Customers: ", train['SalesPerCustomers'].describe())

        st.write("""##### Check possibility of negative Sales in Train CSV File. As a result there are not negative Sales.""")
        st.code("""sales_minus = train[(train["Sales"] < 0)]""")
        sales_minus = train[(train["Sales"] < 0)]
        st.write("Sales Minus: ", sales_minus)

        st.write("""##### Closed stores and days which didn't have any sales won't be counted in Train CSV File.""")
        st.code("""train = train[(train["Open"] != 0) & (train['Sales'] != 0)]""")

        st.write("""##### Are there any missing values in Store CSV File?""")
        st.code("""store.isnull().sum()""")
        st.write("Missing Values: ", store.isnull().sum())

        st.write("""##### Filling Missing Values in Store CSV File.""")
        st.code("""store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])
store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])
store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(
store['CompetitionOpenSinceMonth'].mode().iloc[0])
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(
store['CompetitionOpenSinceYear'].mode().iloc[0])
train.isnull().sum()
train.dropna()
train.isnull().sum()
train.dropna()
test.isnull().sum()
pd.isna(store)
store.dropna()""")
        store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
        store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])
        store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])
        store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
        store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(
        store['CompetitionOpenSinceMonth'].mode().iloc[0])
        store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(
        store['CompetitionOpenSinceYear'].mode().iloc[0])

        train.isnull().sum()
        train.dropna()
        train.isnull().sum()
        train.dropna()
        test.isnull().sum()
        pd.isna(store)
        store.dropna()
        st.write("Filling Missing Values: ", store.head())

        st.write("""##### Merging Train and Store CSV Files based on Store CSV File.Also filling Nan Values.""")
        st.code("""train_store = train.merge(store, on='Store', how='left')
train_store.dropna()""")
        train_store = train.merge(store, on='Store', how='left')
        train_store.dropna()
        st.write(train_store.head())

        st.write("""##### Plot about Sales Per Store based on Store Type each Month. """)
        st.code("""gr_sales_per_mon = sns.catplot(data=train_store, x='Month', y="Sales",
col='StoreType',  # per store type in cols
palette='plasma',
hue='StoreType',
kind='bar',
row='Promo')
st.pyplot(gr_sales_per_mon)""")
        gr_sales_per_mon = sns.catplot(data=train_store, x='Month', y="Sales",
                    col='StoreType',  # per store type in cols
                    palette='plasma',
                    hue='StoreType',
                    kind='bar',
                    row='Promo')
        st.pyplot(gr_sales_per_mon)

        st.write("""##### Plot about Customers Per Store for each Month. """)
        st.code("""gr_cust_per_month = sns.catplot(data=train_store, x='Month', y="Customers",
col='StoreType',  # per store type in cols
palette='plasma',
hue='StoreType',
kind='bar',
row='Promo')
st.pyplot(gr_cust_per_month)""")
        gr_cust_per_month = sns.catplot(data=train_store, x='Month', y="Customers",
                    col='StoreType',  # per store type in cols
                    palette='plasma',
                    hue='StoreType',
                    kind='bar',
                    row='Promo')
        st.pyplot(gr_cust_per_month)

        # # StoreType pie
        st.write("""##### Pie Chart about Sales Percentages Per Store Type""")
        st.code("""store_types = store['StoreType'].value_counts().sort_values(ascending=False)
pie_store_type_sales, ax = plt.subplots()
ax.pie(store_types, autopct="%.1f%%", startangle=90)
ax.set_title('StoreType pie chart')
st.pyplot(pie_store_type_sales)""")
        store_types = store['StoreType'].value_counts().sort_values(ascending=False)
        pie_store_type_sales, ax = plt.subplots()
        # To Do - labels
        # stores_label = list(store['StoreType'])
        ax.pie(store_types, autopct="%.1f%%", startangle=90)
        ax.set_title('StoreType pie chart')
        st.pyplot(pie_store_type_sales)

        # # DayOfWeek vs Sales
        st.write("""##### Sales Per Day""")
        st.code("""""")
        gr_sales_per_day = sns.catplot(data=train, x='DayOfWeek', y="Sales",
                                        col='Sales',  # per store type in cols
                                        palette='plasma',
                                        hue='Sales',
                                        kind='bar',
                                        row='Promo')
        st.pyplot(gr_sales_per_day)

        # Vasili Fix it
        # fig, ax = plt.subplots(figsize=(15, 10))
        # sns.barplot(x="DayOfWeek", y="Sales", data=train)
        #
        # cat_cols = train_store.select_dtypes(include=['object']).columns
        #
        # for i in cat_cols:
        #     st.write(i)
        #     st.write(train_store[i].value_counts())
        #     st.write('-' * 20)
        #
        # st.pyplot(fig)

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
