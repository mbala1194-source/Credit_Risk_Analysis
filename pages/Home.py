import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder



st.title("Credit Risk Analysis")
st.write('This is an uncleaned raw dataframe showed in the agdatagrid.This has been concatenated from two different files.')
df_data_one=pd.read_csv("files/Loan_data_1st_half.csv")
df_data_two=pd.read_json("files/Loan_data_2nd_half.json")
df=pd.concat([df_data_one,df_data_two],axis=1)
gb = GridOptionsBuilder.from_dataframe(df)

gb.configure_default_column(
    filter=True,
    sortable=True,
    resizable=True,
    minWidth=150,
    wrapHeaderText=True,
    autoHeaderHeight=True,
    tooltipField=None
)

# Pagination
gb.configure_pagination(
    enabled=True,
    paginationAutoPageSize=False,
    paginationPageSize=20
)

# Enable filter sidebar
gb.configure_side_bar()

grid_options = gb.build()

# Display grid
AgGrid(
    df,
    gridOptions=grid_options,
    height=600,
    theme='streamlit',
    fit_columns_on_grid_load=False,
    width='100%'
)
st.header("Cleaned Dataframe")
st.subheader('This table contains the cleaned dataset after performing data preprocessing, including duplicate removal, data type corrections, and missing value handling')

cat_columns=df.select_dtypes(include='object').columns

df['NAME_CONTRACT_TYPE'].replace('XNA', np.nan, inplace=True)

df['NAME_PAYMENT_TYPE'] = df['NAME_PAYMENT_TYPE'].replace('XNA', 'Unknown')

# %%
df['NAME_CASH_LOAN_PURPOSE'] = df['NAME_CASH_LOAN_PURPOSE'].replace({
    'XAP': 'Not applicable',
    'XNA': 'Unknown'
})

df['CODE_REJECT_REASON'] = df['CODE_REJECT_REASON'].replace({
    'XAP': 'Not applicable',
    'XNA': 'Unknown'
})

df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].str.replace(', partner','')

categories=['Other_B','Other_A','Group of people']
df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].replace(categories,'others')

df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna('Missing')

df['NAME_CLIENT_TYPE'] = df['NAME_CLIENT_TYPE'].replace({
    'XNA': 'Unknown'})

df['NAME_GOODS_CATEGORY'] = df['NAME_GOODS_CATEGORY'].replace({
    'XNA': 'Unknown'})

df['NAME_PORTFOLIO'] = df['NAME_PORTFOLIO'].replace({
    'XNA': 'Unknown'})

df['NAME_PRODUCT_TYPE']=df['NAME_PRODUCT_TYPE'].replace({'XNA':'Unknown'})

df['NAME_SELLER_INDUSTRY']=df['NAME_SELLER_INDUSTRY'].replace({'XNA':'Unknown'})

df['NAME_YIELD_GROUP']=df['NAME_YIELD_GROUP'].replace({'XNA':'Unknown'})

df['PRODUCT_COMBINATION']=df['PRODUCT_COMBINATION'].replace({'XNA':'Unknown'})

num_columns=df.select_dtypes(exclude='object').columns

#Q1=df['AMT_ANNUITY'].quantile(0.25)
#Q3=df['AMT_ANNUITY'].quantile(0.75)
#IQR=Q3-Q1
#lower_bound=Q1-1.5*IQR
#upper_bound=Q3+1.5*IQR

df['AMT_ANNUITY'].plot(kind='box')

df['AMT_ANNUITY'].plot(kind='hist')

df['AMT_ANNUITY']=df['AMT_ANNUITY'].median()

df['AMT_DOWN_PAYMENT']=pd.to_numeric(df['AMT_DOWN_PAYMENT'],errors='coerce')

pd.set_option('display.float_format', '{:.2f}'.format)

df['AMT_DOWN_PAYMENT']=df['AMT_DOWN_PAYMENT'].fillna(df['AMT_DOWN_PAYMENT'].median())

df['AMT_GOODS_PRICE']=df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())

df['RATE_DOWN_PAYMENT'] = df['RATE_DOWN_PAYMENT'].fillna(
    df['RATE_DOWN_PAYMENT'].median()
)

df.drop(columns=['RATE_INTEREST_PRIMARY'], inplace=True)

df.drop(columns=['RATE_INTEREST_PRIVILEGED'], inplace=True)

df['CNT_PAYMENT']=df['CNT_PAYMENT'].fillna(df['CNT_PAYMENT'].median())

df['DAYS_FIRST_DRAWING'] = df['DAYS_FIRST_DRAWING'].fillna(
    df['DAYS_FIRST_DRAWING'].median())

DAYS_FIRST_DUE_null=(19161/50000)*100

df['DAYS_FIRST_DUE']=df['DAYS_FIRST_DUE'].fillna(df['DAYS_FIRST_DUE'].median())

df['DAYS_LAST_DUE_1ST_VERSION']=df['DAYS_LAST_DUE_1ST_VERSION'].fillna(df['DAYS_LAST_DUE_1ST_VERSION'].median())

df['DAYS_LAST_DUE']=df['DAYS_LAST_DUE'].fillna(df['DAYS_LAST_DUE'].median())

df['DAYS_TERMINATION']=df['DAYS_TERMINATION'].fillna(df['DAYS_TERMINATION'].median())

df['NFLAG_INSURED_ON_APPROVAL']=df['NFLAG_INSURED_ON_APPROVAL'].fillna(df['NFLAG_INSURED_ON_APPROVAL'].mode()[0])

df.drop('SK_ID_CURR',axis=1,inplace=True)

AgGrid(
    df,
    gridOptions=grid_options,
    height=600,
    theme='streamlit',
    fit_columns_on_grid_load=False,
    width='100%'
)
