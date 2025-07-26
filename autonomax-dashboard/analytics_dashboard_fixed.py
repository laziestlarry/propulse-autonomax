import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Analytics Dashboard", layout="wide")

# Sample data
df = pd.DataFrame({
    "Date": pd.date_range(start="2023-01-01", periods=30),
    "Views": pd.Series(range(30)) * 100 + 500,
    "Revenue": pd.Series(range(30)) * 5 + 100
})

st.title("📊 Fiverr & YouTube AI Business Analytics")
st.markdown("Live dashboard powered by AutonomaX")

col1, col2 = st.columns(2)

with col1:
    fig_views = px.line(df, x="Date", y="Views", title="Daily Views")
    st.plotly_chart(fig_views, use_container_width=True)

with col2:
    fig_revenue = px.bar(df, x="Date", y="Revenue", title="Daily Revenue")
    st.plotly_chart(fig_revenue, use_container_width=True)

st.success("Dashboard Ready. Real data sync coming next...")