import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Oil Forecast Project", layout="wide")
st.title("โครงการพยากรณ์ราคาน้ำมัน (เวอร์ชันอ่านง่าย)")

uploaded = st.file_uploader("อัปโหลดไฟล์ CSV ที่นี่", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    # หา column วันที่
    date_col = None
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            date_col = c
            break
        except:
            pass

    df["Year"] = pd.to_datetime(df[date_col]).dt.year

    # หาบริษัททั้งหมด (ยกเว้น date + year)
    companies = [c for c in df.columns if c not in [date_col, "Year"]]

    st.write("บริษัทที่พบ:", companies)

    years_future = st.slider("ทำนายเพิ่ม (ปี)", 1, 10, 3)

    fig, ax = plt.subplots(figsize=(12, 6))

    forecast_all = []

    for comp in companies:

        # เตรียมข้อมูล
        X = df[["Year"]]
        y = df[comp]

        # โมเดล
        model = LinearRegression()
        model.fit(X, y)

        # ทำนายปัจจุบัน
        pred_now = model.predict(X)

        # ทำนายอนาคต
        last_year = df["Year"].max()
        future_years = np.arange(last_year+1, last_year+years_future+1)
        pred_future = model.predict(future_years.reshape(-1, 1))

        # เก็บผล
        forecast_all.append(pd.DataFrame({
            "Company": comp,
            "Year": future_years,
            "Forecast": pred_future
        }))

        # วาดให้ดูง่าย --------------------------
        ax.plot(df["Year"], y, "o", label=f"{comp} (จริง)")      # จุดจริง
        ax.plot(df["Year"], pred_now, "-", alpha=0.6)            # เส้นแนวโน้ม
        ax.plot(future_years, pred_future, "--", label=f"{comp} (อนาคต)")  # อนาคต
        # ---------------------------------------

    ax.set_title("การพยากรณ์ราคาน้ำมัน (ดูง่าย ไม่รก)")
    ax.set_xlabel("ปี")
    ax.set_ylabel("ราคา")

    ax.grid(True, linestyle="--", alpha=0.3)

    # Legend มุมบนขวา
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    # รวมผลลัพธ์
    output_df = pd.concat(forecast_all)
    st.write("ผลการทำนาย")
    st.dataframe(output_df)

else:
    st.info("อัปโหลดไฟล์ก่อนเริ่มทำงานครับ")

