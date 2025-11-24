import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
st.set_page_config(layout='wide', page_title='Startup Funding Analysis')

# -------------------- DATA LOADING & CLEANING -------------------- #
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / 'startup_cleaned.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Clean 'amount' column (handles "₹", ",", "Cr", "crore", etc.)
def clean_amount(x):
    if pd.isna(x):
        return 0
    x = str(x).strip()
    x = x.replace('₹', '').replace(',', '').replace('crore', '').replace('Cr', '').strip()
    try:
        val = float(x)
        return val
    except:
        # try converting lakhs to crores if pattern matches
        if 'L' in x or 'lakh' in x.lower():
            digits = ''.join([c for c in x if c.isdigit() or c == '.'])
            return float(digits) / 100
        return 0

df['amount'] = df['amount'].apply(clean_amount)
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['investors'] = df['investors'].fillna('Unknown')
df['startup'] = df['startup'].fillna('Unknown')
df['vertical'] = df['vertical'].fillna('Unknown')
df['city'] = df['city'].fillna('Unknown')

# Helper: investor exploded dataframe
investor_exploded = df.assign(investor=df['investors'].str.split(',')).explode('investor')
investor_exploded['investor'] = investor_exploded['investor'].str.strip()


# -------------------- OVERALL ANALYSIS -------------------- #
def load_overall_analysis():
    st.title('📊 Overall Funding Analysis')

    total = df['amount'].sum().round(2)
    max_funding = df.groupby('startup')['amount'].sum().max()
    avg_funding = df.groupby('startup')['amount'].sum().mean()
    num_startups = df['startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Funding', f"{total:,.2f} Cr")
    col2.metric('Max Funding (Single Startup)', f"{max_funding:,.2f} Cr")
    col3.metric('Avg Funding per Startup', f"{avg_funding:,.2f} Cr")
    col4.metric('Unique Funded Startups', num_startups)

    st.markdown("---")
    st.header("📅 Month-on-Month Trend")
    selected_option = st.selectbox('Select Type', ['Total Funding', 'Number of Deals'])

    temp_df = (df.groupby(['year', 'month'])['amount'].sum().reset_index()
               if selected_option == 'Total Funding'
               else df.groupby(['year', 'month'])['amount'].count().reset_index())
    temp_df['Period'] = temp_df['month'].astype(str) + '-' + temp_df['year'].astype(str)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(temp_df['Period'], temp_df['amount'], marker='o')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Top charts
    st.markdown("### 🏆 Top 10 Startups by Total Funding")
    top_startups = df.groupby('startup')['amount'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_startups)

    st.markdown("### 💼 Top 10 Investors by Total Funding")
    investor_funds = investor_exploded.groupby('investor')['amount'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(investor_funds)

    st.markdown("### 🌐 Top 5 Sectors by Total Funding")
    top_sectors = df.groupby('vertical')['amount'].sum().sort_values(ascending=False).head(5)
    fig1, ax1 = plt.subplots()
    ax1.pie(top_sectors.values, labels=top_sectors.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig1)


# -------------------- STARTUP ANALYSIS -------------------- #
def load_startup_details(startup):
    st.title(f"🚀 {startup} - Startup Analysis")

    startup_df = df[df['startup'].str.lower() == startup.lower()]
    if startup_df.empty:
        st.warning("No data found for this startup.")
        return

    total = startup_df['amount'].sum()
    rounds = startup_df.shape[0]
    investors = ", ".join(startup_df['investors'].unique())
    city = startup_df['city'].iloc[0]
    vertical = startup_df['vertical'].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Funding", f"{total:,.2f} Cr")
    col2.metric("Funding Rounds", rounds)
    col3.metric("City", city)

    st.subheader(f"Sector: {vertical}")
    st.markdown(f"**Investors:** {investors}")

    st.subheader("Funding Timeline")
    fig, ax = plt.subplots()
    ax.plot(startup_df['date'], startup_df['amount'], marker='o')
    plt.xlabel("Date")
    plt.ylabel("Funding (Cr)")
    st.pyplot(fig)


# -------------------- INVESTOR ANALYSIS -------------------- #
def load_investor_details(investor):
    st.title(f"💰 {investor} - Investor Analysis")

    investor_df = investor_exploded[investor_exploded['investor'].str.lower() == investor.lower()]
    if investor_df.empty:
        st.warning("No data found for this investor.")
        return

    last5_df = investor_df[['date', 'startup', 'vertical', 'city', 'round', 'amount']]\
        .sort_values('date', ascending=False).head(5)
    st.subheader("Most Recent Investments")
    st.dataframe(last5_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Biggest Investments")
        big_series = investor_df.groupby('startup')['amount'].sum().sort_values(ascending=False).head(10)
        if big_series.empty:
            st.info("No investment data available for chart.")
        else:
            st.bar_chart(big_series)

    with col2:
        st.subheader("Sectors Invested In")
        sector_series = investor_df.groupby('vertical')['amount'].sum().sort_values(ascending=False)
        if sector_series.empty:
            st.info("No sector data available for this investor.")
        else:
            fig, ax = plt.subplots()
            ax.pie(sector_series, labels=sector_series.index, autopct='%1.1f%%')
            st.pyplot(fig)

    st.subheader("YoY Investment Trend")
    year_series = investor_df.groupby('year')['amount'].sum()
    if year_series.empty:
        st.info("No yearly investment data available.")
    else:
        fig2, ax2 = plt.subplots()
        ax2.plot(year_series.index, year_series.values, marker='o')
        plt.xlabel("Year")
        plt.ylabel("Investment (Cr)")
        st.pyplot(fig2)


# -------------------- COMPARISON TOOL -------------------- #
def compare_investors(inv1, inv2):
    st.title(f"⚖️ Comparison: {inv1} vs {inv2}")
    df1 = investor_exploded[investor_exploded['investor'].str.lower() == inv1.lower()]
    df2 = investor_exploded[investor_exploded['investor'].str.lower() == inv2.lower()]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(inv1)
        st.metric("Total Investment", f"{df1['amount'].sum():,.2f} Cr")
        st.metric("Unique Startups", df1['startup'].nunique())
    with col2:
        st.subheader(inv2)
        st.metric("Total Investment", f"{df2['amount'].sum():,.2f} Cr")
        st.metric("Unique Startups", df2['startup'].nunique())

    st.subheader("Sector Distribution Comparison")
    col3, col4 = st.columns(2)
    with col3:
        s1 = df1.groupby('vertical')['amount'].sum().sort_values(ascending=False).head(5)
        st.bar_chart(s1)
    with col4:
        s2 = df2.groupby('vertical')['amount'].sum().sort_values(ascending=False).head(5)
        st.bar_chart(s2)


# -------------------- SMART INSIGHTS -------------------- #
def show_insights():
    st.title("💡 Smart Insights Summary")

    latest_year = int(df['year'].dropna().max())
    year_df = df[df['year'] == latest_year]

    top_sector = year_df.groupby('vertical')['amount'].sum().idxmax()
    top_city = year_df.groupby('city')['amount'].sum().idxmax()
    top_investor = investor_exploded.groupby('investor')['amount'].sum().idxmax()

    total_investment = year_df['amount'].sum()
    num_startups = year_df['startup'].nunique()

    st.write(f"In **{latest_year}**, startups raised a total of **₹{total_investment:,.2f} Cr** "
             f"across **{num_startups} startups**. The most funded sector was **{top_sector}**, "
             f"with **{top_city}** leading by city. Top investor of the year was **{top_investor}**.")


# -------------------- SIDEBAR -------------------- #
st.sidebar.title('🏢 Startup Funding Explorer')
option = st.sidebar.selectbox('Select Section', [
    'Overall Analysis',
    'StartUp Analysis',
    'Investor Analysis',
    'Investor Comparison',
    'Smart Insights'
])

if option == 'Overall Analysis':
    load_overall_analysis()

elif option == 'StartUp Analysis':
    selected_startup = st.sidebar.selectbox('Select a Startup', sorted(df['startup'].unique().tolist()))
    if st.sidebar.button('Analyze Startup'):
        load_startup_details(selected_startup)

elif option == 'Investor Analysis':
    all_investors = sorted(investor_exploded['investor'].dropna().unique().tolist())
    selected_investor = st.sidebar.selectbox('Select an Investor', all_investors)
    if st.sidebar.button('Analyze Investor'):
        load_investor_details(selected_investor)

elif option == 'Investor Comparison':
    all_investors = sorted(investor_exploded['investor'].dropna().unique().tolist())
    inv1 = st.sidebar.selectbox('Investor 1', all_investors, key='inv1')
    inv2 = st.sidebar.selectbox('Investor 2', all_investors, key='inv2')
    if st.sidebar.button('Compare'):
        compare_investors(inv1, inv2)

elif option == 'Smart Insights':
    show_insights()

elif option == 'Investor Comparison':
    all_investors = sorted(investor_exploded['investor'].dropna().unique().tolist())
    inv1 = st.sidebar.selectbox('Investor 1', all_investors, key='inv1')
    inv2 = st.sidebar.selectbox('Investor 2', all_investors, key='inv2')
    if st.sidebar.button('Compare'):
        compare_investors(inv1, inv2)

elif option == 'Smart Insights':
    show_insights()
