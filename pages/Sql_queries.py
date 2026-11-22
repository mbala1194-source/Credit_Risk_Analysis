import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder
from sqlalchemy import create_engine

st.title("SQL Queries")

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

cat_columns=df.select_dtypes(include='object').columns

df['NAME_CONTRACT_TYPE'].replace('XNA', np.nan, inplace=True)

df['NAME_PAYMENT_TYPE'] = df['NAME_PAYMENT_TYPE'].replace('XNA', 'Unknown')


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

##################################################################################################################
username='root'
password='1234'
host='localhost'
database='credit_risk_analysis'
engine=create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}",echo=False)
#connection_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

df.to_sql('loan_data',con=engine,if_exists='replace',index=False)

##############################################################################################################

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

st.subheader("1. Basic Queries")


st.text("1.1. Retrieve all records from the loan_data table")
df=pd.read_sql('SELECT * FROM loan_data',con=engine)

# Display grid
AgGrid(
    df,
    gridOptions=grid_options,
    height=600,
    theme='streamlit',
    fit_columns_on_grid_load=False,
    width='100%'
)


st.text("1.2. Display all loan applications with their contract type and contract status")
df=pd.read_sql('SELECT NAME_CONTRACT_TYPE, NAME_CONTRACT_STATUS FROM loan_data',con=engine)

st.write(df)
 
st.text("1.3. Retrieve all applications where the contract status is 'Approved'.")

df=pd.read_sql('SELECT * FROM loan_data WHERE NAME_CONTRACT_STATUS="Approved"',con=engine)

# Display grid
AgGrid(
    df,
    gridOptions=grid_options,
    height=600,
    theme='streamlit',
    fit_columns_on_grid_load=False,
    width='100%'
)


st.text("1.4. Display all applications where the contract status is 'Refused'.")
df=pd.read_sql('SELECT * FROM loan_data  WHERE NAME_CONTRACT_STATUS="Refused"',con=engine)

# Display grid
AgGrid(
    df,
    gridOptions=grid_options,
    height=600,
    theme='streamlit',
    fit_columns_on_grid_load=False,
    width='100%'
)


st.text("1.5. Retrieve all applications belonging to the 'Repeater' client type.")
df=pd.read_sql('SELECT * FROM loan_data WHERE NAME_CLIENT_TYPE="Repeater"',con=engine)

# Display grid
AgGrid(
    df,
    gridOptions=grid_options,
    height=600,
    theme='streamlit',
    fit_columns_on_grid_load=False,
    width='100%'
)

st.subheader("2. Aggregation Queries")

st.text("2.1. Calculate the total number of loan applications.")

cnt=pd.read_sql('SELECT COUNT(*) AS total_loan_applications FROM loan_data',con=engine)

st.write("Count:", cnt)


st.text("2.2. Calculate the total credit amount across all applications.")

df=pd.read_sql('SELECT SUM(AMT_CREDIT) AS total_credit_amount FROM loan_data',con=engine)

st.write(df)

st.text("2.3. Calculate the average loan application amount.")

df=pd.read_sql('SELECT AVG(AMT_APPLICATION) AS average_loan_application FROM loan_data',con=engine)

st.write(df)


st.text("2.4. Count the number of applications for each contract status.")

df=pd.read_sql('SELECT NAME_CONTRACT_STATUS ,COUNT(*) AS total_application FROM loan_data GROUP BY NAME_CONTRACT_STATUS ORDER BY total_application DESC',con=engine)

st.write(df)


st.text("2.5. Calculate the average annuity amount for each contract type.")

df=pd.read_sql('SELECT NAME_CONTRACT_TYPE,AVG(AMT_ANNUITY) AS  avg_amt_annuity FROM loan_data GROUP BY NAME_CONTRACT_TYPE ORDER BY avg_amt_annuity DESC',con=engine)

st.write(df)

st.subheader("3. Join-Based Queries")

st.text("3.1. Retrieve loan applications where the credit amount is greater than 500,000.")

df=pd.read_sql('SELECT * FROM loan_data WHERE AMT_CREDIT>50000',con=engine)

# Display grid
AgGrid(
    df,
    gridOptions=grid_options,
    height=600,
    theme='streamlit',
    fit_columns_on_grid_load=False,
    width='100%'
)

st.text("3.2. Find applications where the down payment amount is greater than 50,000.")

df=pd.read_sql('SELECT * FROM loan_data WHERE AMT_DOWN_PAYMENT>50000',con=engine)

# Display grid
AgGrid(
    df,
    gridOptions=grid_options,
    height=600,
    theme='streamlit',
    fit_columns_on_grid_load=False,
    width='100%'
)

st.text("3.3. Count the number of applications for each client type.")

df=pd.read_sql('SELECT NAME_CLIENT_TYPE,COUNT(*) AS total_applications FROM loan_data GROUP BY NAME_CLIENT_TYPE ORDER BY total_applications DESC',con=engine)

st.write(df)

st.text("3.4. Calculate the average credit amount for each loan contract status")

df=pd.read_sql('SELECT NAME_CONTRACT_STATUS,AVG(AMT_CREDIT) AS avg_credit_amt FROM loan_data GROUP BY NAME_CONTRACT_STATUS ORDER BY avg_credit_amt DESC',con=engine)

st.write(df)

st.text("3.5. Find the top 5 product combinations with the highest number of applications.")

df=pd.read_sql('SELECT PRODUCT_COMBINATION ,COUNT(*) as total_application FROM loan_data GROUP BY PRODUCT_COMBINATION ORDER BY total_application DESC LIMIT 5',con=engine)

st.write(df)

st.subheader("4. Financial Tracking Queries")

st.text("4.1. Calculate the total credit amount for approved and refused applications separately.")

df=pd.read_sql('SELECT NAME_CONTRACT_STATUS,SUM(AMT_CREDIT) AS total_credit_amount FROM loan_data WHERE NAME_CONTRACT_STATUS IN ("Approved","Refused")GROUP BY NAME_CONTRACT_STATUS ORDER BY total_credit_amount DESC',con=engine)

st.write(df)

st.text("4.2. Find the top 3 contract types with the highest average credit amount.")

df=pd.read_sql('SELECT NAME_CONTRACT_TYPE,AVG(AMT_CREDIT) AS avg_credit_amount FROM loan_data GROUP BY NAME_CONTRACT_TYPE ORDER BY avg_credit_amount DESC LIMIT 3 ',con=engine)

st.write(df)

st.text("4.3. Calculate the total application amount for each payment type.")

df=pd.read_sql('SELECT NAME_PAYMENT_TYPE,SUM(AMT_APPLICATION) AS total_application_amt  FROM loan_data GROUP BY NAME_PAYMENT_TYPE ORDER BY total_application_amt DESC',con=engine)

st.write(df)

st.text("4.4. Identify the top 5 seller industries based on total credit amount.")

df=pd.read_sql('SELECT NAME_SELLER_INDUSTRY,SUM(AMT_CREDIT) AS total_credit_amt FROM loan_data GROUP BY NAME_SELLER_INDUSTRY ORDER BY total_credit_amt DESC LIMIT 5',con=engine)

st.write(df)

st.text("4.5. Calculate the approval percentage for each contract type.")

df=pd.read_sql('SELECT NAME_CONTRACT_TYPE,COUNT(*) AS total_application,SUM(CASE WHEN NAME_CONTRACT_STATUS="Approved" THEN 1 ELSE 0 END)AS approved_application,ROUND(SUM(CASE WHEN NAME_CONTRACT_STATUS="Approved" THEN 1 ELSE 0 END)*100/COUNT(*),2)AS approval_percentage FROM loan_data GROUP BY NAME_CONTRACT_TYPE ORDER BY approval_percentage DESC',con=engine)

st.write(df)