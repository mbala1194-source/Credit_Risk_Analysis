import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import sklearn
from scipy.stats import f_oneway
from sqlalchemy import create_engine
from st_aggrid import AgGrid, GridOptionsBuilder



# %%
df_data_one=pd.read_csv("files/Loan_data_1st_half.csv")

# %%
#df_data_one.shape

# %%
#df_data_one

# %%
df_data_two=pd.read_json("files/Loan_data_2nd_half.json")

# %%
#df_data_two.shape

# %%
#df_data_two

# %%
df=pd.concat([df_data_one,df_data_two],axis=1)
st.title("Credit Risk Analysis")
# %%
#df
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



# %%
df.shape

# %%
df.info()

# %%
df.dtypes

# %%
df['NAME_CONTRACT_TYPE']

# %%
df['SK_ID_CURR']

# %%
df.isnull().sum()

# %%
df.isnull().mean()*100

# %%
df.duplicated().sum()

# %%
df.duplicated(subset=['SK_ID_CURR']).sum()

# %%
df.drop_duplicates(subset=['SK_ID_CURR'],inplace=True)

# %%
df.duplicated().sum()

# %%
df.describe()

# %%
df.describe(include=['object'])

# %%
df

# %%
df.select_dtypes(exclude='object').columns

# %%
df.select_dtypes(exclude='object').isnull().sum()

# %%
cat_columns=df.select_dtypes(include='object').columns

# %%
cat_columns

# %%
df['NAME_CONTRACT_TYPE'].value_counts(dropna=False)

# %%
df['NAME_CONTRACT_TYPE'].replace('XNA', np.nan, inplace=True)

# %%
df['NAME_CONTRACT_TYPE'].isnull().mean() * 100

# %%
df['WEEKDAY_APPR_PROCESS_START'].unique()

# %%
df['WEEKDAY_APPR_PROCESS_START'].value_counts()

# %%
df['FLAG_LAST_APPL_PER_CONTRACT'].unique()

# %%
df['FLAG_LAST_APPL_PER_CONTRACT'].value_counts(dropna=False)

# %%
df['NAME_PAYMENT_TYPE'].unique

# %%
df['NAME_PAYMENT_TYPE'].value_counts(dropna=False)

# %%
df['NAME_PAYMENT_TYPE'] = df['NAME_PAYMENT_TYPE'].replace('XNA', 'Unknown')

# %%
df['NAME_PAYMENT_TYPE'].value_counts(dropna=False)

# %%
df['NAME_CASH_LOAN_PURPOSE'].value_counts(dropna=False)

# %%
df['NAME_CASH_LOAN_PURPOSE'] = df['NAME_CASH_LOAN_PURPOSE'].replace({
    'XAP': 'Not applicable',
    'XNA': 'Unknown'
})

# %%
df['NAME_CASH_LOAN_PURPOSE'].isnull().mean()*100

# %%
df['NAME_CONTRACT_STATUS'].unique()

# %%
df['NAME_CONTRACT_STATUS'].value_counts(dropna=False)

# %%
df['CODE_REJECT_REASON'].unique

# %%
df['CODE_REJECT_REASON'].value_counts(dropna=False)

# %%
df['CODE_REJECT_REASON'] = df['CODE_REJECT_REASON'].replace({
    'XAP': 'Not applicable',
    'XNA': 'Unknown'
})

# %%
df['NAME_TYPE_SUITE'].value_counts(dropna=False)

# %%
df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].str.replace(', partner','')

# %%
df[cat_columns].isnull().sum()

# %%
df[cat_columns].isnull().mean()*100

# %%
df['NAME_TYPE_SUITE'].value_counts(dropna=False)

# %%
categories=['Other_B','Other_A','Group of people']
df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].replace(categories,'others')

# %%
df['NAME_TYPE_SUITE'].value_counts(dropna=False)

# %%
df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna('Missing')

# %%
df['NAME_TYPE_SUITE'].value_counts(dropna=False)

# %%
df['NAME_CLIENT_TYPE'].unique

# %%
df['NAME_CLIENT_TYPE'].value_counts(dropna=False)

# %%
df['NAME_CLIENT_TYPE'] = df['NAME_CLIENT_TYPE'].replace({
    'XNA': 'Unknown'})

# %%
df['NAME_CLIENT_TYPE'].value_counts(dropna=False)

# %%
df['NAME_GOODS_CATEGORY'].unique

# %%
df['NAME_GOODS_CATEGORY'].value_counts(dropna=False)

# %%
df['NAME_GOODS_CATEGORY'] = df['NAME_GOODS_CATEGORY'].replace({
    'XNA': 'Unknown'})

# %%
df['NAME_PORTFOLIO'].unique

# %%
df['NAME_PORTFOLIO'].value_counts(dropna=False)

# %%
df['NAME_PORTFOLIO'] = df['NAME_PORTFOLIO'].replace({
    'XNA': 'Unknown'})

# %%
df['NAME_PRODUCT_TYPE'].unique

# %%
df['NAME_PRODUCT_TYPE'].value_counts(dropna=False)

# %%
df['NAME_PRODUCT_TYPE']=df['NAME_PRODUCT_TYPE'].replace({'XNA':'Unknown'})

# %%
df['CHANNEL_TYPE'].unique

# %%
df['CHANNEL_TYPE'].value_counts(dropna=False)

# %%
df['NAME_SELLER_INDUSTRY'].unique

# %%
df['NAME_SELLER_INDUSTRY'].value_counts(dropna=False)

# %%
df['NAME_SELLER_INDUSTRY']=df['NAME_SELLER_INDUSTRY'].replace({'XNA':'Unknown'})

# %%
df['NAME_YIELD_GROUP'].value_counts(dropna=False)

# %%
df['NAME_YIELD_GROUP']=df['NAME_YIELD_GROUP'].replace({'XNA':'Unknown'})

# %%
df['PRODUCT_COMBINATION'].unique

# %%
df['PRODUCT_COMBINATION'].isnull().sum()

# %%
df['PRODUCT_COMBINATION'].value_counts(dropna=False)

# %%
df['PRODUCT_COMBINATION']=df['PRODUCT_COMBINATION'].replace({'XNA':'Unknown'})

# %%
num_columns=df.select_dtypes(exclude='object').columns

# %%
num_columns

# %%
df['NFLAG_INSURED_ON_APPROVAL'].dtype

# %%
df.select_dtypes(exclude='object').isnull().mean()*100

# %%
df['AMT_CREDIT'].describe()

# %%
df['AMT_ANNUITY'].describe()

# %%
Q1=df['AMT_ANNUITY'].quantile(0.25)

Q3=df['AMT_ANNUITY'].quantile(0.75)


# %%
IQR=Q3-Q1

# %%
IQR

# %%
lower_bound=Q1-1.5*IQR

# %%
lower_bound

# %%
upper_bound=Q3+1.5*IQR

# %%
upper_bound

# %%
df['AMT_ANNUITY'].plot(kind='box')

# %%
df['AMT_ANNUITY'].plot(kind='hist')

# %%
df['AMT_ANNUITY']=df['AMT_ANNUITY'].median()

# %%
df['AMT_ANNUITY'].isnull().sum()

# %%
df['AMT_DOWN_PAYMENT'].describe()

# %%
df['AMT_DOWN_PAYMENT'].unique

# %%
df['AMT_DOWN_PAYMENT']=pd.to_numeric(df['AMT_DOWN_PAYMENT'],errors='coerce')

# %%
pd.set_option('display.float_format', '{:.2f}'.format)
df['AMT_DOWN_PAYMENT'].describe()

# %%
df['AMT_DOWN_PAYMENT']=df['AMT_DOWN_PAYMENT'].fillna(df['AMT_DOWN_PAYMENT'].median())

# %%
df['AMT_DOWN_PAYMENT'].isnull().sum()

# %%
df['AMT_GOODS_PRICE'].describe()

# %%
df['AMT_GOODS_PRICE']=df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())

# %%
df['AMT_GOODS_PRICE'].isnull().sum()

# %%
df['RATE_DOWN_PAYMENT'].describe()

# %%
df['RATE_DOWN_PAYMENT'].isnull().mean()*100

# %%
df['RATE_DOWN_PAYMENT'] = df['RATE_DOWN_PAYMENT'].fillna(
    df['RATE_DOWN_PAYMENT'].median()
)

# %%
df['RATE_DOWN_PAYMENT'].isnull().sum()

# %%
df['RATE_INTEREST_PRIMARY'].describe()

# %%
df['RATE_INTEREST_PRIMARY'].isnull().mean()*100

# %%
df.drop(columns=['RATE_INTEREST_PRIMARY'], inplace=True)

# %%
df['RATE_INTEREST_PRIVILEGED'].isnull().mean()*100

# %%
df['RATE_INTEREST_PRIVILEGED'].describe()

# %%
df.drop(columns=['RATE_INTEREST_PRIVILEGED'], inplace=True)

# %%
df['CNT_PAYMENT'].isnull().mean()*100


# %%
df['CNT_PAYMENT'].describe()

# %%
df['CNT_PAYMENT']

# %%
(df['CNT_PAYMENT'] == 0).sum()

# %%
df['CNT_PAYMENT']=df['CNT_PAYMENT'].fillna(df['CNT_PAYMENT'].median())

# %%
df['CNT_PAYMENT'].isnull().sum()

# %%
df['DAYS_FIRST_DRAWING'].isnull().mean()*100

# %%
df['DAYS_FIRST_DRAWING'].describe()

# %%
df['DAYS_FIRST_DRAWING'] = df['DAYS_FIRST_DRAWING'].fillna(
    df['DAYS_FIRST_DRAWING'].median())

# %%
df['DAYS_FIRST_DRAWING'].isnull().sum()

# %%
df['DAYS_FIRST_DUE'].isnull().mean()*100

# %%
df['DAYS_FIRST_DUE'].describe()

# %%
DAYS_FIRST_DUE_null=(19161/50000)*100
DAYS_FIRST_DUE_null

# %%
df['DAYS_FIRST_DUE'].skew()

# %%
df['DAYS_FIRST_DUE']=df['DAYS_FIRST_DUE'].fillna(df['DAYS_FIRST_DUE'].median())

# %%
df['DAYS_LAST_DUE_1ST_VERSION'].isnull().mean()*100

# %%
df['DAYS_LAST_DUE_1ST_VERSION'].describe()

# %%
df['DAYS_LAST_DUE_1ST_VERSION']=df['DAYS_LAST_DUE_1ST_VERSION'].fillna(df['DAYS_LAST_DUE_1ST_VERSION'].median())

# %%
df['DAYS_LAST_DUE_1ST_VERSION'].isnull().sum()

# %%
df['DAYS_LAST_DUE'].isnull().mean()*100

# %%
df['DAYS_LAST_DUE'].describe()

# %%
df['DAYS_LAST_DUE']=df['DAYS_LAST_DUE'].fillna(df['DAYS_LAST_DUE'].median())

# %%
df['DAYS_LAST_DUE'].isnull().sum()

# %%
df['DAYS_TERMINATION'].isnull().mean()*100

# %%
df['DAYS_TERMINATION'].describe()

# %%
df['DAYS_TERMINATION']=df['DAYS_TERMINATION'].fillna(df['DAYS_TERMINATION'].median())

# %%
df['DAYS_TERMINATION'].isnull().sum()

# %%
df['NFLAG_INSURED_ON_APPROVAL'].isnull().mean()*100

# %%
df['NFLAG_INSURED_ON_APPROVAL']

# %%
df['NFLAG_INSURED_ON_APPROVAL'].describe()

# %%
df['NFLAG_INSURED_ON_APPROVAL'].skew()

# %%
df['NFLAG_INSURED_ON_APPROVAL'].median()

# %%
df['NFLAG_INSURED_ON_APPROVAL'].mode()

# %%
df['NFLAG_INSURED_ON_APPROVAL']=df['NFLAG_INSURED_ON_APPROVAL'].fillna(df['NFLAG_INSURED_ON_APPROVAL'].mode()[0])

# %%
df['NFLAG_INSURED_ON_APPROVAL'].isnull().sum()

# %%
df.select_dtypes(exclude='object').isnull().mean()*100

# %%
df.duplicated().sum()

# %%
df.columns

# %%
df.drop('SK_ID_CURR',axis=1,inplace=True)

# %%
df.describe()

# %%
for column in df.select_dtypes(exclude='object').columns[1:]:
  sns.histplot(x=column, data=df,bins=40,kde=True)
  plt.title(f'Distribution of {column}')
  plt.show()

# %%
for column in df.select_dtypes(include='object').columns:
  plt.figure(figsize=(6,4))
  sns.countplot(x=column, data=df)
  plt.title(f'Countplot of {column}')
  plt.show()

# %%
num_columns=df.select_dtypes(exclude='object').columns[1:]
cat_columns=df.select_dtypes(include='object').columns

# %%
cat_columns,len(cat_columns)

# %%
num_columns,len(num_columns)

# %%
num_bivariate_columns = [
    'AMT_APPLICATION',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'AMT_DOWN_PAYMENT',
    'CNT_PAYMENT'
]

# %%
num_num_columns = [
    ('AMT_APPLICATION', 'AMT_CREDIT'),
    ('AMT_GOODS_PRICE', 'AMT_CREDIT'),
    ('AMT_CREDIT', 'CNT_PAYMENT'),
    ('AMT_APPLICATION', 'AMT_DOWN_PAYMENT')
    
]

# %%
for x,y in num_num_columns:
    

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        x=x,
        y=y,
        data=df,
        alpha=0.3
    )

    plt.title(f'{x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

# %%
#The scatter plot of `AMT_APPLICATION` vs `AMT_CREDIT` shows a strong positive correlation, indicating that the approved credit amount generally increases with the requested application amount. Most data points are concentrated along an upward trend, suggesting that financial institutions tend to approve higher credit amounts for applications requesting larger loans. A few outliers are also observed.

# %%
#The scatter plot of `AMT_GOODS_PRICE` vs `AMT_CREDIT` shows a strong positive correlation, indicating that the credit amount generally increases with the price of the goods. Most data points follow a clear upward trend, suggesting that higher-priced goods are associated with higher credit amounts. A few outliers are also observed.

# %%
#he scatter plot of `AMT_CREDIT` vs `CNT_PAYMENT` shows that higher credit amounts are generally associated with a larger number of repayment installments. However, the data points are widely distributed across different installment counts, indicating that repayment duration varies among loan applicants. Most applications are concentrated at lower credit amounts, with installment periods commonly ranging from 6 to 60 months.

# %%
#The scatter plot of `AMT_APPLICATION` vs `AMT_DOWN_PAYMENT` shows a weak positive relationship, indicating that down payment amounts generally increase with the loan application amount. However, many applications have zero down payment, while a few records show unusually high down payment values. This suggests considerable variation in down payment behaviour among loan applicants.

# %%
for x, y in num_num_columns:

    corr = df[x].corr(df[y])

    print(f'Correlation between {x} and {y}: {corr:.2f}')

# %%
correlation = df[num_bivariate_columns].corr()

# %%
plt.figure(figsize=(10, 7))

Correlation = df[num_bivariate_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    square=True
)

plt.title('Correlation Heatmap')
plt.show()

# %%
cat_num_columns = [
    ('NAME_CONTRACT_STATUS', 'AMT_CREDIT'),
    ('NAME_CONTRACT_TYPE', 'AMT_CREDIT'),
    ('NAME_CLIENT_TYPE', 'AMT_APPLICATION'),
    ('NAME_PORTFOLIO', 'AMT_CREDIT'),
    ('NAME_YIELD_GROUP', 'RATE_DOWN_PAYMENT')
]

# %%
for x, y in cat_num_columns:

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        x=x,
        y=y,
        data=df
    )

    plt.title(f'{x} vs {y}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%
for cat_columns, num_columns in cat_num_columns:

    groups = [
        group[num_columns].dropna()
        for _, group in df.groupby(cat_columns)
    ]

    groups = [group for group in groups if len(group) > 0]

    f_stat, p_value = f_oneway(*groups)

    print(f'\n{cat_columns} vs {num_columns}')
    print('F-Statistic:', round(f_stat, 4))
    print('P-value:', round(p_value, 6))

    if p_value < 0.05:
        print('Reject H0: Significant difference between group means.')
    else:
        print('Fail to reject H0: No significant difference between group means.')

# %%
for cat_col, num_col in cat_num_columns:

    print(f'\n{cat_col} vs {num_col}')

    display(
        df.groupby(cat_col)[num_col]
        .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        .round(2)
    )

# %%
#The box plot of `NAME_CONTRACT_STATUS` vs `AMT_CREDIT` shows that refused applications have a higher median credit amount and wider distribution compared to approved applications. Cancelled applications generally have very low credit amounts, while unused offers show relatively lower credit amounts. Several outliers are observed across all categories, indicating variations in credit amounts among loan applications.

# %%
#The box plot of `NAME_CONTRACT_TYPE` vs `AMT_CREDIT` shows that cash loans have the highest median credit amount and the widest distribution compared to consumer and revolving loans. Consumer loans have comparatively lower median credit amounts, while revolving loans show moderate credit amounts. Several extreme outliers are observed across all contract types, indicating significant variations in credit amounts among different loan categories.

# %%
#The box plot of `NAME_CLIENT_TYPE` vs `AMT_APPLICATION` shows that repeat clients have a higher median application amount compared to new clients. Refreshed clients also show relatively higher application amounts, while new clients have a comparatively lower median. Several extreme outliers are observed across all categories, indicating significant variations in loan application amounts among different client types.

# %%
#The box plot of `NAME_PORTFOLIO` vs `AMT_CREDIT` shows that car loan applications have the highest median credit amount, followed by cash and card loans. POS loans have comparatively lower median credit amounts, while unknown portfolio categories show very low credit amounts. Several outliers are observed, particularly in cash and POS portfolios, indicating significant variations in credit amounts across different loan portfolios.

# %%
#The box plot of `NAME_YIELD_GROUP` vs `RATE_DOWN_PAYMENT` shows that the middle, high and low-normal yield groups have relatively similar median down payment rates. The low-action group shows a comparatively lower median, while the unknown category has very limited variation. Several outliers are observed across the yield groups, indicating differences in down payment behaviour among loan applicants.


# %%
cat_cat_columns = [
    ('NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS'),
    ('NAME_CLIENT_TYPE', 'NAME_CONTRACT_STATUS'),
    ('NAME_PORTFOLIO', 'NAME_CONTRACT_STATUS'),
    ('NAME_PAYMENT_TYPE', 'NAME_CONTRACT_STATUS'),
    ('CHANNEL_TYPE', 'NAME_CONTRACT_STATUS')
]

# %%
for x, y in cat_cat_columns:

    ct = pd.crosstab(df[x], df[y])

    ct.plot(
        kind='bar',
        stacked=True,
        figsize=(9, 5)
    )

    plt.title(f'{x} vs {y}')
    plt.xlabel(x)
    plt.ylabel('Application Count')
    plt.xticks(rotation=45)
    plt.legend(title=y)
    plt.tight_layout()
    plt.show()




# %%

ct = pd.crosstab(df[x], df[y], normalize='index') * 100

# %%
#The relationship between **NAME_CONTRACT_TYPE** and **NAME_CONTRACT_STATUS** shows that consumer loans have the highest number of applications, with the majority being approved, followed by a smaller number of refused and unused offers. Cash loans have a comparatively higher proportion of cancelled and refused applications, while revolving loans have the lowest overall application count. This indicates that consumer loans have a higher approval count, whereas cash loans show a greater number of unsuccessful or cancelled applications.

# %%
#The relationship between **NAME_CLIENT_TYPE** and **NAME_CONTRACT_STATUS** shows that repeat clients account for the highest number of previous loan applications, followed by new and refreshed clients. Among repeat clients, approved applications are the highest, although a considerable number of applications were also refused or cancelled. New and refreshed clients have comparatively fewer applications, with approvals forming the largest share in both categories. This indicates that repeat clients contribute significantly to the previous loan application records and account for the highest number of approved applications.

# %%
#The relationship between **NAME_PORTFOLIO** and **NAME_CONTRACT_STATUS** shows that POS accounts for the highest number of previous loan applications, with the majority being approved, followed by a smaller number of refused applications. Cash portfolio applications also show a considerable number of approvals and refusals. The Cards category has fewer applications, while the Unknown category has a higher number of cancelled applications and unused offers. This indicates that POS and Cash portfolios contribute significantly to approved loan applications, whereas the Unknown category has a comparatively higher number of cancelled applications.

# %%
#The relationship between **NAME_PAYMENT_TYPE** and **NAME_CONTRACT_STATUS** shows that applications using 'Cash through the bank' have the highest number of records, with the majority being approved, followed by refused applications. Applications with an 'Unknown' payment type have a considerable number of cancelled applications, along with some refused applications. Other payment methods account for very few applications. This indicates that bank-based cash payments are the most common payment method among previous loan applications, while applications with unknown payment types show a higher proportion of cancellations.

# %%
#The relationship between **CHANNEL_TYPE** and **NAME_CONTRACT_STATUS** shows that Country-wide and Credit and Cash Offices account for the highest number of previous loan applications. Country-wide applications have a higher number of approvals, whereas Credit and Cash Offices show a considerable number of cancelled and refused applications. Regional / Local and Stone channels have comparatively fewer applications, with approvals forming the majority of their records. This indicates that loan application outcomes vary across different distribution channels, with Credit and Cash Offices recording a relatively higher number of cancelled applications.

# %%
username='root'
password='1234'
host='localhost'
database='credit_risk_analysis'
engine=create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}",echo=False)
#connection_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

df.to_sql('loan_data',con=engine,if_exists='replace',index=False)

# %%
#basic queries

# %%
df=pd.read_sql('SELECT * FROM loan_data',con=engine)

# %%
df

# %%
#Retrieve all records from the loan_data table.


# %%
df=pd.read_sql('SELECT * FROM loan_data WHERE NAME_CONTRACT_TYPE="Cash loans" ',con=engine)

# %%
df

# %%


#Display all loan applications with their contract type and contract status






# %%
df=pd.read_sql('SELECT NAME_CONTRACT_TYPE AND NAME_CONTRACT_STATUS FROM loan_data',con=engine)

# %%
df

# %%
#Retrieve all applications where the contract status is 'Approved'.



# %%
df=pd.read_sql('SELECT * FROM loan_data WHERE NAME_CONTRACT_STATUS="Approved"',con=engine)
df

# %%
#Display all applications where the contract status is 'Refused'.


# %%
df=pd.read_sql('SELECT * FROM loan_data  WHERE NAME_CONTRACT_STATUS="Refused"',con=engine)
df

# %%
#Retrieve all applications belonging to the 'Repeater' client type.


# %%
df=pd.read_sql('SELECT * FROM loan_data WHERE NAME_CLIENT_TYPE="Repeater"',con=engine)
df

# %% [markdown]
# #🔹 Aggregation Queries
# 

# %%
#Calculate the total number of loan applications.


# %%
df=pd.read_sql('SELECT COUNT(*) AS total_loan_applications FROM loan_data',con=engine)
df

# %%
#Calculate the total credit amount across all applications.


# %%
df=pd.read_sql('SELECT SUM(AMT_CREDIT) AS total_credit_amount FROM loan_data',con=engine)
df

# %%
#Calculate the average loan application amount.


# %%
df=pd.read_sql('SELECT AVG(AMT_APPLICATION) AS average_loan_application FROM loan_data',con=engine)
df

# %%
#Count the number of applications for each contract status.


# %%
df=pd.read_sql('SELECT NAME_CONTRACT_STATUS ,COUNT(*) AS total_application FROM loan_data GROUP BY NAME_CONTRACT_STATUS ORDER BY total_application DESC',con=engine)
df

# %%
#Calculate the average annuity amount for each contract type.

# %%
df=pd.read_sql('SELECT NAME_CONTRACT_TYPE,AVG(AMT_ANNUITY) AS  avg_amt_annuity FROM loan_data GROUP BY NAME_CONTRACT_TYPE ORDER BY avg_amt_annuity DESC',con=engine)
df

# %%
#🔹 Join-Based Queries


# %%
#Retrieve loan applications where the credit amount is greater than 500,000.


# %%
df=pd.read_sql('SELECT * FROM loan_data WHERE AMT_CREDIT>50000',con=engine)
df

# %%
#Find applications where the down payment amount is greater than 50,000.


# %%
df=pd.read_sql('SELECT * FROM loan_data WHERE AMT_DOWN_PAYMENT>50000',con=engine)
df

# %%
#Count the number of applications for each client type.


# %%
df=pd.read_sql('SELECT NAME_CLIENT_TYPE,COUNT(*) AS total_applications FROM loan_data GROUP BY NAME_CLIENT_TYPE ORDER BY total_applications DESC',con=engine)
df

# %%
#Calculate the average credit amount for each loan contract status.


# %%
df=pd.read_sql('SELECT NAME_CONTRACT_STATUS,AVG(AMT_CREDIT) AS avg_credit_amt FROM loan_data GROUP BY NAME_CONTRACT_STATUS ORDER BY avg_credit_amt DESC',con=engine)
df

# %%
#Find the top 5 product combinations with the highest number of applications.


# %%
df=pd.read_sql('SELECT PRODUCT_COMBINATION ,COUNT(*)as total_application FROM loan_data GROUP BY PRODUCT_COMBINATION ORDER BY total_application DESC',con=engine)
df

# %%
#🔹 Financial Tracking Queries


# %%
#Calculate the total credit amount for approved and refused applications separately.


# %%
df=pd.read_sql('SELECT NAME_CONTRACT_STATUS,SUM(AMT_CREDIT) AS total_credit_amount FROM loan_data WHERE NAME_CONTRACT_STATUS IN ("Approved","Refused")GROUP BY NAME_CONTRACT_STATUS ORDER BY total_credit_amount DESC',con=engine)
df

# %%
#Find the top 3 contract types with the highest average credit amount.


# %%
df=pd.read_sql('SELECT NAME_CONTRACT_TYPE,AVG(AMT_CREDIT) AS avg_credit_amount FROM loan_data GROUP BY NAME_CONTRACT_TYPE ORDER BY avg_credit_amount DESC LIMIT 3 ',con=engine)
df

# %%
#Calculate the total application amount for each payment type.


# %%
df=pd.read_sql('SELECT NAME_PAYMENT_TYPE,SUM(AMT_APPLICATION) AS total_application_amt  FROM loan_data GROUP BY NAME_PAYMENT_TYPE ORDER BY total_application_amt DESC',con=engine)
df

# %%
#Identify the top 5 seller industries based on total credit amount.


# %%
df=pd.read_sql('SELECT NAME_SELLER_INDUSTRY,SUM(AMT_CREDIT) AS total_credit_amt FROM loan_data GROUP BY NAME_SELLER_INDUSTRY ORDER BY total_credit_amt DESC LIMIT 5',con=engine)
df



# %%
#Calculate the approval percentage for each contract type.


# %%
df=pd.read_sql('SELECT NAME_CONTRACT_TYPE,COUNT(*) AS total_application,SUM(CASE WHEN NAME_CONTRACT_STATUS="Approved" THEN 1 ELSE 0 END)AS approved_application,ROUND(SUM(CASE WHEN NAME_CONTRACT_STATUS="Approved" THEN 1 ELSE 0 END)*100/COUNT(*),2)AS approval_percentage FROM loan_data GROUP BY NAME_CONTRACT_TYPE ORDER BY approval_percentage DESC',con=engine)
df


