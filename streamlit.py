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
from run_models import predict


def main():

    # Dimiourgia menu
    menu = ["Introduction", "Data Exploration", "Models", "Conclusion"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Introduction":
        st.write("""# Data Science Assignment
# Sales Prediction""")
        st.info("Students names: Vasilis Andreou, Prodromos Pieri, Polyxeni Xerou")
        st.write("""# Introduction""")
        st.subheader("""Problem""")
        st.write(""" Using the available data (historical sales data) to follow the necessary steps and develop a model where he can predict the sales that can be made in the future (in the relevant test dataset).""")
        with st.expander("Files"):
            st.write("""
            train.csv - historical data including Sales
            
            test.csv - historical data excluding Sales
            
            store.csv - supplemental information about the stores""")

        with st.expander("Data fields"):
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
from run_models import predict
import xgboost as xgb
import streamlit as st
from run_models import predict""")

        #Diavasma twn csv Arxeiwn
        train = pd.read_csv("train.csv", low_memory=False)
        st.write("##### Reading Train CSV File ", train.head())
        st.code("train = pd.read_csv('train.csv', low_memory=False)")

        store = pd.read_csv("store.csv")
        st.write("##### Reading Store CSV File ", store.head())
        st.code("store = pd.read_csv('store.csv')")

        test = pd.read_csv("test.csv")
        st.write("##### Reading Test CSV File ", test.head())
        st.code("test = pd.read_csv('test.csv')")

        st.write("##### Reading and Splitting Date into Year, Month, Day Columns in Train CSV File. ")
        st.code("""train['Date'] = pd.to_datetime(train['Date'])
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day""")
        ## Allagi tou datatype
        train['Date'] = pd.to_datetime(train['Date'])
        # data extraction
        train['Year'] = train['Date'].dt.year
        train['Month'] = train['Date'].dt.month
        train['Day'] = train['Date'].dt.day
        st.write("Result after splitting Date: ", train.head())

        st.write("##### Calculate the average of Sales Per Customers in Train CSV File.")
        train['SalesPerCustomers'] = train['Sales'] / train['Customers']  # prosthiki neas metavlitis
        st.code("train['SalesPerCustomers'].describe()")
        st.write("Average of Sales Per Customers: ", train['SalesPerCustomers'].describe())
        st.write("##### Observation")
        st.write("On average customers spend about 9.50$ per day.")

        st.write("""##### Check possibility of negative Sales in Train CSV File.""")
        st.code("""sales_minus = train[(train["Sales"] < 0)]""")
        sales_minus = train[(train["Sales"] < 0)]
        st.write("Sales Minus: ", sales_minus)
        st.write("##### Observation")
        st.write("There are not negative Sales.")

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
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])
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
        store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])
        store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])

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
        st.write("##### Observations")
        st.write("Each store has more sales in duration of promo and the last month of the year stores with promo make more sales.")

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
        st.write("##### Observations")
        st.write("Each store attracts more customers and makes more sales in duration of promo.")
        # # StoreType pie
        st.write("""##### Pie Chart about Sales Percentages Per Store Type""")
        st.code("""store_types = store['StoreType'].value_counts().sort_values(ascending=False)
pie_store_type_sales, ax = plt.subplots()
st.write(store_types)
ax.pie(store_types, autopct="%.1f%%", startangle=90, labels = store_types.keys())
ax.set_title('StoreType pie chart')
st.pyplot(pie_store_type_sales)""")
        store_types = store['StoreType'].value_counts().sort_values(ascending=False)
        pie_store_type_sales, ax = plt.subplots()
        st.write(store_types)
        ax.pie(store_types, autopct="%.1f%%", startangle=90, labels = store_types.keys())
        ax.set_title('StoreType pie chart')
        st.pyplot(pie_store_type_sales)
        st.write("##### Observations")
        st.write("""There are 4 type of stores(a,b,c,d).
Stores of type a has higher amount of total Customers and Sales.
StoreType d goes on the second place in both Sales and Customers.""")
        # # DayOfWeek vs Sales
        st.write("""##### Sales Per Day""")
        st.code("""fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x="DayOfWeek", y="Sales", data=train)
st.pyplot(fig))""")
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.barplot(x="DayOfWeek", y="Sales", data=train)
        st.pyplot(fig)
        st.write("##### Observations")
        st.write("""Sales and Customers are both very less on Sundays as most of the stores are closed on Sunday.
So, Sales on Monday is highest in whole week. 
This might be due to the fact that stores are closed on Sundays.""")

        cat_cols = train_store.select_dtypes(include=['object']).columns
        for i in cat_cols:
            st.write(i)
            st.write(train_store[i].value_counts())
            st.write('-' * 20)

        # Plot average sales & customers with/without promo
        st.write("""##### Plot average sales and customers with and without promo.""")
        st.code("""fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
sns.barplot(x='Promo', y='Sales', data=train_store, ax=axis1)
sns.barplot(x='Promo', y='Customers', data=train_store, ax=axis2)
st.pyplot(fig)""")
        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
        sns.barplot(x='Promo', y='Sales', data=train_store, ax=axis1)
        sns.barplot(x='Promo', y='Customers', data=train_store, ax=axis2)
        st.pyplot(fig)
        st.write("##### Observations")
        st.write("""Both Sales and Customers increases by a significant amount during Promotions.
This shows that Promotion has a positive effect for a store.""")

        # Plot CompetitionDistance Vs Sales
        st.write("""##### Plot CompetitionDistance Vs Sales""")
        st.code("""gr_competitors, ax = plt.subplots()
sns.scatterplot(data=train_store, x ='CompetitionDistance',y='Sales')
st.pyplot(gr_competitors)""")
        gr_competitors, ax = plt.subplots()
        sns.scatterplot(data=train_store, x ='CompetitionDistance',y='Sales')
        st.pyplot(gr_competitors)
        st.write("##### Observations")
        st.write("""Most of the stores have their competition within 5km range, that is too close.
The closer the competitors are, the more sales there are.""")

        # Barplots for average sales and customers with or without promo
        st.write("""##### Plots for average sales and customers on state holidays""")
        st.code("""gr_holidays, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
sns.barplot(x='StateHoliday', y='Sales', data=train_store, ax=axis1)
sns.barplot(x='StateHoliday', y='Customers', data=train_store, ax=axis2)
st.pyplot(gr_holidays)""")
        gr_holidays, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
        sns.barplot(x='StateHoliday', y='Sales', data=train_store, ax=axis1)
        sns.barplot(x='StateHoliday', y='Customers', data=train_store, ax=axis2)
        st.pyplot(gr_holidays)
        st.write("##### Observations")
        st.write("""Most of the stores remain closed during State and School Holidays.
But it is interesting to note that the number of stores opened during School Holidays were more than that were opened during State Holidays.
Another important thing to note is that the stores which were opened during School holidays had more sales than normal.""")

        # Barplot for average sales and customers on school holidays
        st.write("""##### Plot for average sales and customers on school holidays""")
        st.code("""sns.countplot(x='SchoolHoliday', data=train_store)
gr_schoolHoliday, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
sns.barplot(x='SchoolHoliday', y='Sales', data=train_store, ax=axis1)
sns.barplot(x='SchoolHoliday', y='Customers', data=train_store, ax=axis2)
st.pyplot(gr_schoolHoliday)""")
        sns.countplot(x='SchoolHoliday', data=train_store)
        gr_schoolHoliday, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
        sns.barplot(x='SchoolHoliday', y='Sales', data=train_store, ax=axis1)
        sns.barplot(x='SchoolHoliday', y='Customers', data=train_store, ax=axis2)
        st.pyplot(gr_schoolHoliday)
        st.write("##### Observations")
        st.write("""Most of the stores remain closed during State and School Holidays.
But it is interesting to note that the number of stores opened during School Holidays were more than that were opened during State Holidays.
Another important thing to note is that the stores which were opened during School holidays had more sales than normal.""")

        # Plots for Assortment and & Assortment Vs average sales and customers
        st.write("""##### Plots for Assortment and & Assortment Vs average sales and customers""")
        st.code("""sns.countplot(x='Assortment', data=train_store, order=['a', 'b', 'c'])
gr_assortment, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
sns.barplot(x='Assortment', y='Sales', data=train_store, order=['a', 'b', 'c'], ax=axis1)
sns.barplot(x='Assortment', y='Customers', data=train_store, order=['a', 'b', 'c'], ax=axis2)
st.pyplot(gr_assortment)""")
        sns.countplot(x='Assortment', data=train_store, order=['a', 'b', 'c'])
        gr_assortment, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
        sns.barplot(x='Assortment', y='Sales', data=train_store, order=['a', 'b', 'c'], ax=axis1)
        sns.barplot(x='Assortment', y='Customers', data=train_store, order=['a', 'b', 'c'], ax=axis2)
        st.pyplot(gr_assortment)
        st.write("##### Observations")
        st.write("""There are 3 types of assortments(a,b,c).
Assortment b has higher sales and customers than the other assortments.""")
        # Plots for Promo2 Vs average sales and customers
        st.write("""##### Plots for Promo2 Vs average sales and customers""")
        st.code("""sns.countplot(x='Promo2', data=train_store)
plots_promo2, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
sns.barplot(x='Promo2', y='Sales', data=train_store, ax=axis1)
sns.barplot(x='Promo2', y='Customers', data=train_store, ax=axis2)
st.pyplot(plots_promo2)""")
        sns.countplot(x='Promo2', data=train_store)
        plots_promo2, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
        sns.barplot(x='Promo2', y='Sales', data=train_store, ax=axis1)
        sns.barplot(x='Promo2', y='Customers', data=train_store, ax=axis2)
        st.pyplot(plots_promo2)
        st.write("##### Observations")
        st.write("""It's a continuous and sequential promotion for some stores: 0 = the store does not participate, 1 = the store participates.
Despite the fact that some stores have continuous and consecutive discounts there are not more sales and customers.""")

        # # Data Manipulation
        st.write("""##### Data Manipulation""")
        st.code("""train_store['StateHoliday'] = train_store['StateHoliday'].map({'0': 0, 0: 0, 'a': 1, 'b': 2, 'c': 3})
train_store['StateHoliday'] = train_store['StateHoliday'].astype('int', errors='ignore')
train_store['StoreType'] = train_store['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
train_store['StoreType'] = train_store['StoreType'].astype(int)
train_store['Assortment'] = train_store['Assortment'].map({'a': 1, 'b': 2, 'c': 3})
train_store['Assortment'] = train_store['Assortment'].astype(int)
train_store['PromoInterval'] = train_store['PromoInterval'].map({'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3})
train_store['PromoInterval'] = train_store['PromoInterval'].astype(int)""")
        train_store['StateHoliday'] = train_store['StateHoliday'].map({'0': 0, 0: 0, 'a': 1, 'b': 2, 'c': 3})
        type_StateHoliday = train_store['StateHoliday'] = train_store['StateHoliday'].astype('int', errors='ignore')
        train_store['StoreType'] = train_store['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
        type_StoreType = train_store['StoreType'] = train_store['StoreType'].astype(int)
        train_store['Assortment'] = train_store['Assortment'].map({'a': 1, 'b': 2, 'c': 3})
        type_Assortment = train_store['Assortment'] = train_store['Assortment'].astype(int)
        train_store['PromoInterval'] = train_store['PromoInterval'].map({'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3})
        type_PromoInterval = train_store['PromoInterval'] = train_store['PromoInterval'].astype(int)

        st.caption("Type of train_store['StateHoliday']")
        st.write(type_StateHoliday.dtypes)
        st.write("")
        st.caption("Type of train_store['StoreType']")
        st.write(type_StoreType.dtypes)
        st.write("")
        st.caption("Type of train_store['Assortment']")
        st.write(type_Assortment.dtypes)
        st.write("")
        st.caption("Type of train_store['PromoInterval']")
        st.write(type_PromoInterval.dtypes)

    elif choice == "Models":
        st.write("""# Models""")
        st.write("Define training and testing sets")
        st.code("""X = train_store.drop(['Sales','Date','Customers','SalesPerCustomers'],1)
y = np.log(train_store['Sales']+1)
from sklearn.model_selection import train_test_split
X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.20 , random_state = 1 )""")

        st.write("""# Choose Model""")
        st.write("""Decision Tree Regressor
        Decision tree is one of the well known and powerful supervised machine learning algorithms that can be used for classification and regression problems. Scikit-learn API provides the DecisionTreeRegressor class to apply decision tree method for regression task.""")
        st.code("""dt = DecisionTreeRegressor(max_depth=11)
dt.fit(X_train , y_train)
y_pred_dt = dt.predict(X_val)

y_pred_dt = np.exp(y_pred_dt)-1
y_val = np.exp(y_val)-1""")
        if st.button("run Decision Tree Regressor"):
            model = 1
            # print(predict(model))
            st.write("Sale Prediction for this model:")
            st.write(predict(model))
        st.write("""An AdaBoost regressor. An AdaBoost regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional 
        copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction.""")
        st.code("""adaboost_tree = AdaBoostRegressor(DecisionTreeRegressor())
adaboost_tree.fit(X_train, y_train)

y_hat = adaboost_tree.predict(X_val)

y_hat = np.exp(y_hat) - 1
y_val = np.exp(y_val) - 1""")

        if st.button("run AdaBoost regressor"):
            model = 2
            st.write(predict(model))
        st.write("""XGBoost stands for "Extreme Gradient Boosting" and it is an implementation of gradient boosting trees algorithm. 
The XGBoost is a popular supervised machine learning model with characteristics like computation speed, parallelization, and performance.""")
        st.code("""import xgboost as xgb
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
y_val = np.exp(y_val) - 1

xgbr = xgb.XGBRegressor(verbosity=0)
xgbr.fit(X_train, y_train)""")
        if st.button("run XGBRegressor"):
            model = 3
            st.write(predict(model))

        st.write("""A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a 
technique called Bootstrap and Aggregation, commonly known as bagging. Random Forest has multiple decision trees as base learning models.""")
        st.code("""randomForest = RandomForestRegressor(n_estimators=25, n_jobs=-1, verbose=1)
randomForest.fit(X_train, y_train)

y_pred_rfd = randomForest.predict(X_val)

y_pred_rfd = np.exp(y_pred_rfd) - 1
y_val = np.exp(y_val) - 1""")
        if st.button("run RandomForestRegressor"):
            model = 4
            st.write(predict(model))

    elif choice == "Conclusion":
        st.write("""# Conclusion""")
        with st.expander("Conclusions"):
            st.write("""
The conclusions came out after we ran the 4 algorithms. 
We noticed that these algorithms allow us to see which variables were most useful and dramatically affected sales forecasts.
The most important of these was the CompetitionDistance variable, ie the distance that stores had from their competitors.
The highest quality algorithm with the highest score was AdaBoostRegressor, followed by RandomForestRegressor, DecisionTreeRegressor and XGBoostRegressor.
It is worth noting that sales forecasts were almost the same as in previous years. Comparatively always per year.""")


if __name__ == '__main__':
    main()

