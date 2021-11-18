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
    st.write("""# Data Science Assignment""")
    menu = ["CSV Files", "Data Exploration", "Models", "Conclusion"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "CSV Files":
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