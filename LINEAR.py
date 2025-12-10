import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Oil Price Forecast", layout="wide", page_icon="⛽")

st.title("⛽📈 Oil Price Forecast — Multi Company")
st.write("พยากรณ์ราคาน้ำมัน 4 บริษัทด้วย Linear Regression")

uploaded_file = st.file_uploader("📌 อัปโหลดไฟล์ CSV (รูปแบบคอลัมน์: Date, WTI, Brent, OPEC, Dubai)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 ตารางข้อมูลราคาน้ำมัน")
    st.dataframe(df)

    df['Year'] = pd.to_datetime(df['Date']).dt.year

    companies = ['WTI', 'Brent', 'OPEC', 'Dubai']

    n_years = st.slider("ต้องการทำนายอนาคตกี่ปี", 1, 10, 3)

    fig, ax = plt.subplots(figsize=(12,6))

    forecast_list = []

    for comp in companies:
        X = df[['Year']]
        y = df[comp]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        last_year = df['Year'].max()
        future_years = np.arange(last_year+1, last_year+n_years+1).reshape(-1,1)
        future_pred = model.predict(future_years)

        future_df = pd.DataFrame({
            "Company": comp,
            "Year": future_years.flatten(),
            "Predicted Price": future_pred
        })
        forecast_list.append(future_df)

        ax.scatter(df['Year'], y, s=100, label=f"{comp} Actual")
        ax.plot(df['Year'], y_pred, '--', label=f"{comp} Trend")
        ax.scatter(future_years, future_pred, marker="^", s=140, label=f"{comp} Forecast")

    result_all = pd.concat(forecast_list)

    st.subheader("🔮 ผลการพยากรณ์ราคาน้ำมัน")
    st.dataframe(result_all.reset_index(drop=True))

    ax.set_title("Oil Price Forecast Comparison")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price (USD per barrel)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    csv = result_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 ดาวน์โหลดผลการพยากรณ์ (CSV)",
        data=csv,
        file_name="forecast_results.csv",
        mime="text/csv"
    )

else:
    st.info("⬆ กรุณาอัปโหลดไฟล์ CSV ก่อน")