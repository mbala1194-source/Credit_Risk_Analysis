import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder
from scipy import stats
from scipy.stats import f_oneway


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

st.title("Bivariate Analysis")
num_bivariate_columns = [
    'AMT_APPLICATION',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'AMT_DOWN_PAYMENT',
    'CNT_PAYMENT'
]
num_num_columns = [
    ('AMT_APPLICATION', 'AMT_CREDIT'),
    ('AMT_GOODS_PRICE', 'AMT_CREDIT'),
    ('AMT_CREDIT', 'CNT_PAYMENT'),
    ('AMT_APPLICATION', 'AMT_DOWN_PAYMENT')
    
]
st.subheader("Numerical vs Numerical Analysis")
chart_num_num_descriptions = {
    'AMT_APPLICATION_AMT_CREDIT': "The scatter plot of `AMT_APPLICATION` vs `AMT_CREDIT` shows a strong positive correlation, indicating that the approved credit amount generally increases with the requested application amount. Most data points are concentrated along an upward trend, suggesting that financial institutions tend to approve higher credit amounts for applications requesting larger loans. A few outliers are also observed.",
    'AMT_GOODS_PRICE_AMT_CREDIT': "The scatter plot of `AMT_GOODS_PRICE` vs `AMT_CREDIT` shows a strong positive correlation, indicating that the credit amount generally increases with the price of the goods. Most data points follow a clear upward trend, suggesting that higher-priced goods are associated with higher credit amounts. A few outliers are also observed.",
    'AMT_CREDIT_CNT_PAYMENT': "The scatter plot of `AMT_CREDIT` vs `CNT_PAYMENT` shows that higher credit amounts are generally associated with a larger number of repayment installments. However, the data points are widely distributed across different installment counts, indicating that repayment duration varies among loan applicants. Most applications are concentrated at lower credit amounts, with installment periods commonly ranging from 6 to 60 months.",
    'AMT_APPLICATION_AMT_DOWN_PAYMENT': "The scatter plot of `AMT_APPLICATION` vs `AMT_DOWN_PAYMENT` shows a weak positive relationship, indicating that down payment amounts generally increase with the loan application amount. However, many applications have zero down payment, while a few records show unusually high down payment values. This suggests considerable variation in down payment behaviour among loan applicants.",
}

for x, y in num_num_columns:

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=x,
        y=y,
        data=df,
        alpha=0.3,
        ax=ax
    )

    ax.set_title(f'{x} vs {y}')
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.markdown("**Insight:**")

    st.write(
    chart_num_num_descriptions.get(
        (x+"_"+y),
        f"Shows the relationship between {x} and {y}."
    )
)
    
    
st.subheader("Numerical Correlation Analysis")

# Display correlation between selected numerical pairs
st.markdown("### Pairwise Correlation")

for x, y in num_num_columns:

    corr = df[x].corr(df[y])

    st.write(f"**Correlation between {x} and {y}:** {corr:.2f}")



st.markdown("### Correlation Heatmap")

correlation = df[num_bivariate_columns].corr()

fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    square=True,
    ax=ax
)

ax.set_title('Correlation Heatmap')

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)
############################################################
cat_num_columns = [
    ('NAME_CONTRACT_STATUS', 'AMT_CREDIT'),
    ('NAME_CONTRACT_TYPE', 'AMT_CREDIT'),
    ('NAME_CLIENT_TYPE', 'AMT_APPLICATION'),
    ('NAME_PORTFOLIO', 'AMT_CREDIT'),
    ('NAME_YIELD_GROUP', 'RATE_DOWN_PAYMENT')
]

chart_cat_num_descriptions = {
    'NAME_CONTRACT_STATUS_AMT_CREDIT': 'The box plot of `NAME_CONTRACT_STATUS` vs `AMT_CREDIT` shows that refused applications have a higher median credit amount and wider distribution compared to approved applications. Cancelled applications generally have very low credit amounts, while unused offers show relatively lower credit amounts. Several outliers are observed across all categories, indicating variations in credit amounts among loan applications.',
    'NAME_CONTRACT_TYPE_AMT_CREDIT': 'The box plot of `NAME_CONTRACT_TYPE` vs `AMT_CREDIT` shows that cash loans have the highest median credit amount and the widest distribution compared to consumer and revolving loans. Consumer loans have comparatively lower median credit amounts, while revolving loans show moderate credit amounts. Several extreme outliers are observed across all contract types, indicating significant variations in credit amounts among different loan categories.',    'AMT_GOODS_PRICE': 'The distribution of AMT_GOODS_PRICE is highly positively skewed. Most loan applications are associated with relatively lower goods prices, while a small proportion have significantly higher valuesThe distribution of AMT_GOODS_PRICE is highly positively skewed. Most loan applications are associated with relatively lower goods prices, while a small proportion have significantly higher values',
    'NAME_CLIENT_TYPE_AMT_APPLICATION': 'The box plot of `NAME_CLIENT_TYPE` vs `AMT_APPLICATION` shows that repeat clients have a higher median application amount compared to new clients. Refreshed clients also show relatively higher application amounts, while new clients have a comparatively lower median. Several extreme outliers are observed across all categories, indicating significant variations in loan application amounts among different client types.',
    'NAME_PORTFOLIO_AMT_CREDIT': 'The box plot of `NAME_PORTFOLIO` vs `AMT_CREDIT` shows that car loan applications have the highest median credit amount, followed by cash and card loans. POS loans have comparatively lower median credit amounts, while unknown portfolio categories show very low credit amounts. Several outliers are observed, particularly in cash and POS portfolios, indicating significant variations in credit amounts across different loan portfolios.',
    'NAME_YIELD_GROUP_RATE_DOWN_PAYMENT': 'The box plot of `NAME_YIELD_GROUP` vs `RATE_DOWN_PAYMENT` shows that the middle, high and low-normal yield groups have relatively similar median down payment rates. The low-action group shows a comparatively lower median, while the unknown category has very limited variation. Several outliers are observed across the yield groups, indicating differences in down payment behaviour among loan applicants.',
}
st.subheader("Categorical vs Numerical Analysis")

for x, y in cat_num_columns:

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        x=x,
        y=y,
        data=df,
        ax=ax
    )

    ax.set_title(f'{x} vs {y}')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.markdown("**Insight:**")

    st.write(
    chart_cat_num_descriptions.get(
        (x, y),
        f"Compares the distribution of {y} across different categories of {x}."
    )
)

st.subheader("ANOVA – Statistical Analysis")

for cat_col, num_col in cat_num_columns:

    groups = [
        group[num_col].dropna()
        for _, group in df.groupby(cat_col)
    ]

    groups = [group for group in groups if len(group) > 0]

    if len(groups) < 2:
        st.warning(f"Not enough groups for {cat_col} vs {num_col}")
        continue

    f_stat, p_value = f_oneway(*groups)

    st.markdown(f"### {cat_col} vs {num_col}")

    col1, col2 = st.columns(2)

    col1.metric("F-Statistic", round(f_stat, 4))
    col2.metric("P-value", f"{p_value:.6f}")

    if p_value < 0.05:
        st.success(
            "Reject H0: Significant difference between group means."
        )
    else:
        st.info(
            "Fail to reject H0: No significant difference between group means."
        )

    st.divider()

    st.subheader("Categorical vs Numerical – Statistical Summary")

for cat_col, num_col in cat_num_columns:

    st.markdown(f"### {cat_col} vs {num_col}")

    summary = (
        df.groupby(cat_col)[num_col]
        .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        .round(2)
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    ###########################################################
    cat_cat_columns = [
    ('NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS'),
    ('NAME_CLIENT_TYPE', 'NAME_CONTRACT_STATUS'),
    ('NAME_PORTFOLIO', 'NAME_CONTRACT_STATUS'),
    ('NAME_PAYMENT_TYPE', 'NAME_CONTRACT_STATUS'),
    ('CHANNEL_TYPE', 'NAME_CONTRACT_STATUS')
    ]

    chart_cat_cat_descriptions = {
    'NAME_CONTRACT_TYPE_NAME_CONTRACT_STATUS': 'The relationship between **NAME_CONTRACT_TYPE** and **NAME_CONTRACT_STATUS** shows that consumer loans have the highest number of applications, with the majority being approved, followed by a smaller number of refused and unused offers. Cash loans have a comparatively higher proportion of cancelled and refused applications, while revolving loans have the lowest overall application count. This indicates that consumer loans have a higher approval count, whereas cash loans show a greater number of unsuccessful or cancelled applications.',
    'NAME_CLIENT_TYPE_NAME_CONTRACT_STATUS': 'The relationship between **NAME_CLIENT_TYPE** and **NAME_CONTRACT_STATUS** shows that repeat clients account for the highest number of previous loan applications, followed by new and refreshed clients. Among repeat clients, approved applications are the highest, although a considerable number of applications were also refused or cancelled. New and refreshed clients have comparatively fewer applications, with approvals forming the largest share in both categories. This indicates that repeat clients contribute significantly to the previous loan application records and account for the highest number of approved applications.',
    'NAME_PORTFOLIO_NAME_CONTRACT_STATUS': 'The relationship between **NAME_PORTFOLIO** and **NAME_CONTRACT_STATUS** shows that POS accounts for the highest number of previous loan applications, with the majority being approved, followed by a smaller number of refused applications. Cash portfolio applications also show a considerable number of approvals and refusals. The Cards category has fewer applications, while the Unknown category has a higher number of cancelled applications and unused offers. This indicates that POS and Cash portfolios contribute significantly to approved loan applications, whereas the Unknown category has a comparatively higher number of cancelled applications.',
    'NAME_PAYMENT_TYPE_NAME_CONTRACT_STATUS': 'The relationship between **NAME_PAYMENT_TYPE** and **NAME_CONTRACT_STATUS** shows that applications using "Cash through the bank" have the highest number of records, with the majority being approved, followed by refused applications. Applications with an "Unknown" payment type have a considerable number of cancelled applications, along with some refused applications. Other payment methods account for very few applications. This indicates that bank-based cash payments are the most common payment method among previous loan applications, while applications with unknown payment types show a higher proportion of cancellations.',
    'CHANNEL_TYPE_NAME_CONTRACT_STATUS': 'The relationship between **CHANNEL_TYPE** and **NAME_CONTRACT_STATUS** shows that Country-wide and Credit and Cash Offices account for the highest number of previous loan applications. Country-wide applications have a higher number of approvals, whereas Credit and Cash Offices show a considerable number of cancelled and refused applications. Regional / Local and Stone channels have comparatively fewer applications, with approvals forming the majority of their records. This indicates that loan application outcomes vary across different distribution channels, with Credit and Cash Offices recording a relatively higher number of cancelled applications.',
}

    st.subheader("Categorical vs Categorical Analysis")

for x, y in cat_cat_columns:

    ct = pd.crosstab(df[x], df[y])

    fig, ax = plt.subplots(figsize=(9, 5))

    ct.plot(
        kind='bar',
        stacked=True,
        ax=ax
    )

    ax.set_title(f'{x} vs {y}')
    ax.set_xlabel(x)
    ax.set_ylabel('Application Count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title=y)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    ct = pd.crosstab(df[x], df[y], normalize='index') * 100
