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
import streamlit as st


def main():
    menu = ["Introduction", "CSV Files", "Data Exploration", "Models", "Conclusion"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Introduction":
        st.write("""# Data Science Assignment""")
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
    elif choice == "CSV Files":
        st.subheader("CSV Files")
        csv_file_train = st.file_uploader("Upload Train File")
        if csv_file_train is not None:
            try:
                train = pd.read_csv(csv_file_train, low_memory=False)
                st.write(train.head())
            except Exception as e:
                print(e)
        csv_file_test = st.file_uploader("Upload Test File", type=["csv"])
        if csv_file_test is not None:
            try:
                test = pd.read_csv(csv_file_test)
                st.write(test.head())
            except Exception as e:
                print(e)
        csv_file_store = st.file_uploader("Upload Store File", type=["csv"])
        if csv_file_store is not None:
            try:
                store = pd.read_csv(csv_file_store)
                st.write(store.head())
            except Exception as e:
                print(e)
    elif choice == "Data Exploration":
        st.subheader("Data Exploration")
    elif choice == "Models":
        st.subheader("Models")
    elif choice == "Conclusion":
        st.subheader("Conclusion")


if __name__ == '__main__':
    main()