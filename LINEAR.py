import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Oil Forecast Project", layout="wide")

st.title("โครงการพยากรณ์ราคาน้ำมัน ")
st.write("ทำโดยนิสิต : ตัวอย่างโปรเจควิชา MTH202")

st.write("**อัปโหลดไฟล์ CSV ที่มีข้อมูลราคาน้ำมันรายปี**")
st.write("ไฟล์ต้องมีคอลัมน์วันที่ แล้วก็ราคาน้ำมันหลายๆบริษัท")

uploaded = st.file_uploader("อัปโหลดไฟล์ CSV ที่นี่", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)
    st.write("**ข้อมูลที่อ่านได้จากไฟล์**")
    st.dataframe(df)

    # หาคอลัมน์ที่เป็นวันที่
    date_col = None
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            date_col = c
            break
        except:
            pass

    if date_col is None:
        st.error("หา column วันที่ไม่เจอครับ ลองเช็คไฟล์ CSV อีกที")
        st.stop()

    st.write("คอลัมน์วันที่คือ:", date_col)

    df["Year"] = pd.to_datetime(df[date_col]).dt.year

    # คอลัมน์บริษัทน้ำมัน
    companies = [c for c in df.columns if c not in [date_col, "Year"]]

    st.write("บริษัทที่พบในไฟล์:", companies)

    years_future = st.slider("ต้องการทำนายเพิ่มอีกกี่ปี?", 1, 10, 3)

    fig, ax = plt.subplots(figsize=(12, 6))
    results = []

    for comp in companies:
        X = df[["Year"]]
        y = df[comp]

        # โมเดลง่ายๆ Linear Regression
        model = LinearRegression()
        model.fit(X, y)

        # ค่าที่มีอยู่จริง
        pred_now = model.predict(X)

        # ที่ทำนายได้
        last = df["Year"].max()
        future_years = np.arange(last+1, last+years_future+1)
        pred_future = model.predict(future_years.reshape(-1, 1))

        # เก็บผล
        tmp = pd.DataFrame({
            "Company": comp,
            "Year": future_years,
            "Forecast": pred_future
        })
        results.append(tmp)

        # วาดกราฟ
        ax.plot(df["Year"], y, "o-", label=f"{comp} (จริง)")
        ax.plot(df["Year"], pred_now, "--", label=f"{comp} (แนวโน้ม)")
        ax.plot(future_years, pred_future, "x-", label=f"{comp} (อนาคต)")

    ax.set_title("กราฟพยากรณ์ราคาน้ำมันแบบง่ายๆ")
    ax.set_xlabel("ปี")
    ax.set_ylabel("ราคา")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)

    # รวมผลทุกบริษัท
    df_out = pd.concat(results)
    st.write("ผลการพยากรณ์")
    st.dataframe(df_out)

    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("ดาวน์โหลดผลลัพธ์ (CSV)", csv, "forecast_output.csv")

else:
    st.info("กรุณาอัปโหลดไฟล์ก่อนเริ่มทำงานครับ")
