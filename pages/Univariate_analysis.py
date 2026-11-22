import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder


st.title("Univariate Analysis")

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

st.subheader("Numerical Distribution Analysis")

chart_descriptions = {
    'AMT_APPLICATION': 'The distribution of AMT_APPLICATION is highly positively skewed. Most loan applications are concentrated at lower application amounts, while a relatively small number of applications have significantly higher amounts.',
    'AMT_CREDIT': 'The distribution of AMT_CREDIT is highly positively skewed. Most customers have relatively smaller credit amounts, while a small proportion have significantly higher credit amounts',
    'AMT_GOODS_PRICE': 'The distribution of AMT_GOODS_PRICE is highly positively skewed. Most loan applications are associated with relatively lower goods prices, while a small proportion have significantly higher valuesThe distribution of AMT_GOODS_PRICE is highly positively skewed. Most loan applications are associated with relatively lower goods prices, while a small proportion have significantly higher values',
    'AMT_DOWN_PAYMENT': 'The distribution of AMT_DOWN_PAYMENT is extremely positively skewed. Most customers have zero or relatively small down payment amounts.',

'HOUR_APPR_PROCESS_START': 'The distribution of HOUR_APPR_PROCESS_START is approximately bell-shaped.Application activity is relatively low during early morning and late evening hours, indicating that most loan applications are processed during regular business hours.',
    'NFLAG_LAST_APPL_IN_DAY': 'The distribution of NFLAG_LAST_APPL_IN_DAY is  having a value of 1 and very few observations having a value of 0. This indicates that most loan applications are marked as the last application processed on a particular day.',
    'RATE_DOWN_PAYMENT': 'The distribution of RATE_DOWN_PAYMENT is positively skewed.A small proportion of applications have higher down payment rates.',
    'DAYS_DECISION': 'The distribution of DAYS_DECISION is negatively skewed. This indicates that most previous loan decisions occurred recently, while some customers have a longer credit application history.',

    'SELLERPLACE_AREA': 'The distribution of SELLERPLACE_AREA is extremely positively skewed, with most observations concentrated near zero and a very small proportion having exceptionally large values.',
        'CNT_PAYMENT': 'The distribution of CNT_PAYMENT is right-skewed, suggesting that most customers have shorter repayment tenures, while fewer customers have longer repayment periods.',
        'DAYS_FIRST_DRAWING': 'The distribution of DAYS_FIRST_DRAWING is right skewed',
        'DAYS_FIRST_DUE': 'The distribution of DAYS_FIRST_DUE is highly right-skewed, indicating that most first due dates around smaller day values',
    'DAYS_LAST_DUE_1ST_VERSION': 'The distribution of DAYS_LAST_DUE_1ST_VERSION is highly around smaller day values, indicating that most loan repayment schedules have relatively short durations.',
        'DAYS_LAST_DUE': 'The distribution of DAYS_LAST_DUE is highly concentrated around smaller day values, indicating that most previous loan repayments have relatively short durations.',
        'DAYS_TERMINATION': 'The distribution of DAYS_TERMINATION is highly around smaller day values, indicating that most previous loan termination dates occur within a relatively shorter period.',
        'NFLAG_INSURED_ON_APPROVAL': 'The distribution of NFLAG_INSURED_ON_APPROVAL shows that most loan applications are not insured, while a smaller proportion have insurance coverage.'
}


for column in df.select_dtypes(include='number').columns[1:]:

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=column,
        bins=40,
        kde=True,
        ax=ax
    )

    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("**Insight:**")
    st.write(chart_descriptions.get(
        column,
        "Distribution of the selected numerical variable."
    ))

    st.divider()

#######################################################################################################################
st.subheader("Categorical Distribution Analysis")
chart_cat_descriptions = {
    'NAME_CONTRACT_TYPE': "The distribution of **NAME_CONTRACT_TYPE** shows that consumer loans account for the highest number of previous loan applications, followed by cash loans. Revolving loans have the lowest number of applications, indicating that consumer and cash loans are more commonly preferred by customers than revolving loans.",
    'WEEKDAY_APPR_PROCESS_START': "The distribution of **WEEKDAY_APPR_PROCESS_START** is fairly uniform across most days of the week, with Wednesday and Friday recording slightly higher application counts. Sunday has the lowest number of applications, indicating that loan applications are processed more frequently on weekdays than on Sundays.",
    'FLAG_LAST_APPL_PER_CONTRACT': "The distribution of **FLAG_LAST_APPL_PER_CONTRACT** shows that the majority of previous loan applications are marked as 'Y', while only a very small proportion are marked as 'N'. This indicates that most records represent the last application made by a customer for a particular contract.",
    'NAME_CASH_LOAN_PURPOSE': "The distribution of **NAME_CASH_LOAN_PURPOSE** shows that the majority of previous loan applications are classified as Not applicable', followed by 'Repairs'. The remaining loan purposes have significantly fewer applications, indicating that most applicants either have no specific cash loan purpose recorded or apply for loans mainly for repair-related needs.",
    'NAME_CONTRACT_STATUS': "The distribution of **NAME_CONTRACT_STATUS** shows that the majority of previous loan applications were approved, while refused and cancelled applications account for considerably smaller proportions. Unused offers represent the lowest number of applications.",
    'NAME_PAYMENT_TYPE': "The distribution of **NAME_PAYMENT_TYPE** shows that most previous loan applications used the 'Cash through the bank' payment method, followed by applications with an 'Unknown' payment type. Other payment methods, such as non-cash transactions and payments through the employer's account, account for only a very small proportion. This indicates that bank-based cash payments are the most commonly recorded payment method among previous loan applications.",
     'CODE_REJECT_REASON': "The distribution of **CODE_REJECT_REASON** shows that the majority of previous loan applications are classified as 'Not applicable', followed by rejection reason 'HC' and 'LIMIT'. Other rejection reasons, such as 'CLIENT', 'SCOFR' and 'SCO', have considerably fewer records. This indicates that a specific rejection reason is not recorded for most applications, while 'HC' and 'LIMIT' are among the more frequently recorded rejection reasons.",
    'NAME_TYPE_SUITE': "The distribution of **NAME_TYPE_SUITE** shows that most previous loan applications have missing information regarding the applicant's accompanying person, followed by applicants who were unaccompanied. Applications where the applicant was accompanied by family members are comparatively fewer, while spouse, children and other categories have lower counts. This indicates that most applicants either applied alone or did not provide information about their accompanying person.",
    'NAME_CLIENT_TYPE': "The distribution of NAME_CLIENT_TYPE shows that the majority of previous loan applications belong to repeat clients, followed by new clients. Refreshed clients account for a smaller proportion, while unknown client types represent a very small number of applications.",
    'NAME_GOODS_CATEGORY': "The distribution of **NAME_GOODS_CATEGORY** shows that 'Mobile' is the most frequently recorded goods category, followed by 'Computers' and 'Audio/Video'. The remaining goods categories have considerably fewer applications. This indicates that previous loan applications are concentrated in a few major goods categories, with mobile-related purchases being the most common.",
    'NAME_PORTFOLIO': "The distribution of **NAME_PORTFOLIO** shows that POS accounts for the highest number of previous loan applications, followed by Cash and Unknown categories. Cards represent a smaller proportion, while Cars have very few applications. This indicates that most previous loan applications are associated with POS and Cash portfolios, whereas card and car-related loans are comparatively less common.",
    'NAME_PRODUCT_TYPE': "The distribution of **NAME_PRODUCT_TYPE** shows that the majority of previous loan applications fall under the 'Unknown' category, followed by 'X-Sell'. The 'Walk-in' category accounts for the lowest number of applications. This indicates that product type information is not specified for a large proportion of applications, while X-Sell is more common than Walk-in among the identified product types.",
    'CHANNEL_TYPE': "The distribution of **CHANNEL_TYPE** shows that 'Credit and Cash Offices' account for the highest number of previous loan applications, followed by 'Country-wide' channels. Regional and other channels contribute comparatively fewer applications, with some channels recording very low counts. This indicates that most previous loan applications were processed through Credit and Cash Offices and Country-wide channels.",
    'NAME_SELLER_INDUSTRY': "The distribution of **NAME_SELLER_INDUSTRY** shows that a few seller industries account for the majority of previous loan applications. One category has the highest number of applications, followed by two other categories with considerably lower counts. The remaining industries contribute only a small proportion, indicating that previous loan applications are concentrated among a limited number of seller industries.",
    'NAME_YIELD_GROUP': "The distribution of **NAME_YIELD_GROUP** shows that the 'Unknown' category has the highest number of previous loan applications, followed by the 'Middle' and 'High' yield groups. The 'Low-Normal' group has a slightly lower count, while the 'Low-Action' group has the fewest applications. This indicates that a considerable proportion of previous loan applications do not have a specified yield group.",
    'PRODUCT_COMBINATION': "The distribution of **PRODUCT_COMBINATION** shows that a few product combinations account for the majority of previous loan applications, while several other combinations have considerably lower counts. The most frequently occurring product combinations have significantly higher application counts than the remaining categories. This indicates that customers tend to prefer certain loan product combinations over others."
}


for column in df.select_dtypes(include='object').columns:

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=column,
        ax=ax
    )

    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("**Insight:**")
    st.write(chart_cat_descriptions.get(
        column,
        "Distribution of the selected categorical variable."
    ))

    st.divider()

