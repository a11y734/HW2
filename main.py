# ============================
# 🚗 Car Price Prediction Project
# Dataset: hellbuoy/car-price-prediction
# Web App Deployment with Streamlit
# ============================

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 解決 matplotlib 中文顯示問題
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題


@st.cache_data
def load_data():
    """
    載入並快取資料，執行基本前處理。
    """
    # 建立相對於目前 .py 檔案的檔案路徑
    script_dir = Path(__file__).parent
    data_file = script_dir / "CarPrice_Assignment.csv"
    df = pd.read_csv(data_file)
    
    # 刪除明顯不必要欄位
    df_processed = df.drop(["car_ID", "CarName"], axis=1)
    
    # 處理類別變數
    categorical_cols = df_processed.select_dtypes(include='object').columns
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, dtype=int)
    
    return df, df_processed

@st.cache_resource
def train_model(df_processed):
    """
    使用 RFE 選擇特徵並訓練線性迴歸模型。
    """
    X = df_processed.drop("price", axis=1)
    y = df_processed["price"]

    # 分割訓練與測試集 (雖然此處用全部資料訓練以獲得更穩定的模型，但保留分割邏輯以供參考)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特徵選擇 (RFE)
    lr = LinearRegression()
    # 選擇15個特徵以包含更多使用者可調整的選項
    rfe = RFE(lr, n_features_to_select=15)
    rfe.fit(X_train, y_train)

    selected_features = X_train.columns[rfe.support_]

    # 使用選出的特徵訓練最終模型
    final_model = LinearRegression()
    final_model.fit(X_train[selected_features], y_train)
    
    # 評估模型以供顯示
    y_pred = final_model.predict(X_test[selected_features])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # 建立特徵重要性 DataFrame
    importance_df = pd.DataFrame({'feature': selected_features, 'importance': np.abs(final_model.coef_)}).sort_values('importance', ascending=False)
    
    return final_model, selected_features, X_train, y_train, (r2, rmse), importance_df, (X_test, y_test)

# ==============================
# Streamlit App UI
# ==============================

st.set_page_config(page_title="🚗 車價預測器", layout="wide")
st.title("Car Price Prediction Dataset")
st.write("""
本專案使用 Kaggle 上的汽車價格資料集，透過多元線性迴歸模型進行分析。
我們利用遞歸特徵消除（RFE）技術自動篩選出影響車價的關鍵特徵，並建立此互動式儀表板。
預期效益是讓使用者能透過調整參數直觀地了解各項特徵對車價的影響，並獲得一個合理的價格預估。
""")

# 載入資料與訓練模型
df_raw, df_processed = load_data()
model, selected_features, X_train, y_train, metrics, importance_df, test_data = train_model(df_processed)

def user_input_features(importance_df, container):
    """
    根據特徵重要性排序，建立使用者輸入介面。
    container: The Streamlit container to draw the widgets in (e.g., st or st.sidebar)
    """
    inputs = {}
    # 建立一個 widget 函數的對應字典
    widget_map = {
        'enginesize': lambda: container.slider('引擎大小 (enginesize)', int(df_raw['enginesize'].min()), int(df_raw['enginesize'].max()), int(df_raw['enginesize'].mean())),
        'curbweight': lambda: container.slider('車重 (curbweight)', int(df_raw['curbweight'].min()), int(df_raw['curbweight'].max()), int(df_raw['curbweight'].mean())),
        'horsepower': lambda: container.slider('馬力 (horsepower)', int(df_raw['horsepower'].min()), int(df_raw['horsepower'].max()), int(df_raw['horsepower'].mean())),
        'carwidth': lambda: container.slider('車寬 (carwidth)', float(df_raw['carwidth'].min()), float(df_raw['carwidth'].max()), float(df_raw['carwidth'].mean())),
        'carlength': lambda: container.slider('車長 (carlength)', float(df_raw['carlength'].min()), float(df_raw['carlength'].max()), float(df_raw['carlength'].mean())),
        'wheelbase': lambda: container.slider('軸距 (wheelbase)', float(df_raw['wheelbase'].min()), float(df_raw['wheelbase'].max()), float(df_raw['wheelbase'].mean())),
        'citympg': lambda: container.slider('城市油耗 (citympg)', int(df_raw['citympg'].min()), int(df_raw['citympg'].max()), int(df_raw['citympg'].mean())),
        'highwaympg': lambda: container.slider('高速油耗 (highwaympg)', int(df_raw['highwaympg'].min()), int(df_raw['highwaympg'].max()), int(df_raw['highwaympg'].mean())),
        'boreratio': lambda: container.slider('缸徑比 (boreratio)', float(df_raw['boreratio'].min()), float(df_raw['boreratio'].max()), float(df_raw['boreratio'].mean())),
        'aspiration': lambda: container.selectbox('進氣方式 (aspiration)', df_raw['aspiration'].unique()),
        'enginelocation': lambda: container.selectbox('引擎位置 (enginelocation)', df_raw['enginelocation'].unique()),
        'enginetype': lambda: container.selectbox('引擎類型 (enginetype)', df_raw['enginetype'].unique()),
        'carbody': lambda: container.selectbox('車體 (carbody)', df_raw['carbody'].unique()),
        'cylindernumber': lambda: container.selectbox('汽缸數 (cylindernumber)', df_raw['cylindernumber'].unique()),
    }
    
    # 根據重要性排序來動態生成 UI
    for feature_name in importance_df['feature']:
        # 處理 one-hot 編碼的特徵，找到原始的特徵名稱
        base_feature = feature_name.split('_')[0]
        if base_feature in widget_map and base_feature not in inputs:
            inputs[base_feature] = widget_map[base_feature]()
    
    # 將類別輸入轉換為 one-hot encoding
    if 'enginetype' in inputs: inputs.update({f'enginetype_{et}': 1 if et == inputs['enginetype'] else 0 for et in df_raw['enginetype'].unique() if et != 'dohc'})
    if 'carbody' in inputs: inputs.update({f'carbody_{cb}': 1 if cb == inputs['carbody'] else 0 for cb in df_raw['carbody'].unique() if cb != 'convertible'})
    if 'cylindernumber' in inputs: inputs.update({f'cylindernumber_{cn}': 1 if cn == inputs['cylindernumber'] else 0 for cn in df_raw['cylindernumber'].unique() if cn != 'four'})
    if 'aspiration' in inputs: inputs.update({'aspiration_turbo': 1 if inputs['aspiration'] == 'turbo' else 0}) # drop_first='std'
    if 'enginelocation' in inputs: inputs.update({'enginelocation_rear': 1 if inputs['enginelocation'] == 'rear' else 0}) # drop_first='front'

    # 建立一個包含所有可能特徵的 DataFrame
    feature_df = pd.DataFrame([inputs])
    # 確保所有模型需要的欄位都存在
    for col in X_train.columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
            
    return feature_df[selected_features], inputs # 同時回傳模型輸入和原始使用者輸入

# --- 主頁面 Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 單一特徵迴歸分析",
    "🚀 特徵重要性與評估指標", 
    "💰 預測車價",
    "📈 整體模型預測效果",
])

with tab4: # 📈 整體模型預測效果
    st.subheader("多元線性迴歸：實際價格 vs. 預測價格")
    st.write("這張圖展示了我們訓練出的多元線性迴歸模型在測試資料上的表現。點越靠近紅色的虛線，代表模型的預測越準確。")
    
    X_test, y_test = test_data
    X_test_selected = X_test[selected_features]
    y_pred_test = model.predict(X_test_selected)

    # 使用 statsmodels 來計算預測區間
    X_test_const = sm.add_constant(X_test_selected)
    X_train_const = sm.add_constant(X_train[selected_features])
    ols_model = sm.OLS(y_train, X_train_const).fit()
    predictions_summary = ols_model.get_prediction(X_test_const).summary_frame(alpha=0.05)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # 再次設定字體以確保在 Streamlit Cloud 上正確顯示
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 為了正確繪製區間，我們需要根據預測值對所有相關資料進行排序
    plot_data = predictions_summary.join(y_test).sort_values('mean')

    ax.scatter(y_pred_test, y_test, alpha=0.5, label="實際值 vs. 預測值")
    ax.fill_between(plot_data['mean'], plot_data['obs_ci_lower'], plot_data['obs_ci_upper'], color='lightblue', alpha=0.4, label='95% 預測區間')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="完美預測線")
    ax.set_xlabel("預測價格 (Predicted Price)")
    ax.set_ylabel("實際價格 (Actual Price)")
    ax.set_title("整體模型預測效果")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.info("""
    **圖表說明：**
    - **藍色散佈點**：代表測試資料中，每一輛車的「預測價格」與「實際價格」的對應關係。
    - **紅色虛線**：是「完美預測線」。如果模型的預測完全準確，所有的點都會落在這條線上。
    - **淺藍色區域**：是 **95% 預測區間 (Prediction Interval)**。這表示對於任何一個新的預測，我們有 95% 的信心認為其「真實價格」會落在这个區間內。
    """, icon="💡")

with tab2: # 🚀 特徵重要性與評估指標
    st.subheader("模型如何「自動」選擇特徵？")
    st.write("我們的模型使用了遞歸特徵消除（RFE, Recursive Feature Elimination）技術，這是一種自動化的特徵篩選方法。它會反覆建立模型，並移除最不重要的特徵，直到剩下指定數量的最佳特徵組合。")
    
    st.subheader("特徵重要性排序")
    st.write("下圖顯示了最終被模型選中的特徵，以及它們各自對車價的影響力（迴歸係數）。")
    
    # 使用原始係數（包含正負號）來繪圖
    importance_series = pd.Series(model.coef_, index=selected_features).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_series.index, importance_series.values)
    ax.set_xlabel("迴歸係數大小 (Coefficient Magnitude)")
    ax.set_title("特徵重要性排序")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    st.info("""
    **圖表解讀：**
    - **正值 (長條向右)**：表示該特徵數值越高，預測的車價也越高。
    - **負值 (長條向左)**：表示該特徵數值越高，預測的車價反而越低。
    - **長條長度**：代表影響程度的大小。
    """, icon="💡")
    
    st.subheader("模型表現指標")
    r2, rmse = metrics
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("R² 分數 (R-squared)", f"{r2:.4f}", help="模型對資料的解釋能力，越接近1越好。")
    m_col2.metric("均方根誤差 (RMSE)", f"${rmse:,.2f}", help="預測值與實際值的平均差異，越小越好。")

with tab1: # 🔍 單一特徵迴歸分析
    st.subheader("單純線性迴歸：探索單一特徵與價格的關係")
    st.write("您可以選擇一個數值特徵，觀察它與車價之間的關係以及迴歸趨勢線。")
    
    # 讓使用者選擇要視覺化的特徵
    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    # 排除已知的非特徵欄位
    features_to_plot = [col for col in numeric_cols if col not in ['car_ID', 'symboling', 'price']]
    selected_feature_for_plot = st.selectbox("選擇一個特徵進行分析：", options=features_to_plot, index=features_to_plot.index('enginesize'))

    fig, ax = plt.subplots()
    sns.regplot(x=df_raw[selected_feature_for_plot], y=df_raw['price'], ax=ax, scatter_kws={'alpha':0.4})
    
    ax.set_title(f"{selected_feature_for_plot} vs. Price")
    ax.set_xlabel(selected_feature_for_plot)
    ax.set_ylabel("Price")
    ax.grid(True)
    st.pyplot(fig)
    
    st.info("""
    **圖表說明：**
    - **灰色散佈點**：代表資料集中每一輛車的原始數據。
    - **藍色實線**：是基於所有數據計算出的單純線性迴歸趨勢線，表示該特徵與價格的大致關係。
    """, icon="💡")

with tab3: # 💰 預測車價
    st.subheader("調整車輛參數以進行預測")
    
    col1, col2 = st.columns(2)
    with col1:
        input_df, user_raw_inputs = user_input_features(importance_df, st)
    
    with col2:
        st.write("#### 預測結果")
        # --- 顯示預測結果 ---
        prediction = model.predict(input_df)
        final_price = max(0, prediction[0]) # 確保價格不為負

        st.metric(label="預測車價", value=f"${final_price:,.2f}")

        if prediction[0] < 0:
            st.warning("注意：模型預測出負價格。這通常表示您選擇的參數組合在現實市場中極為罕見或不存在。雖然輸入值都在合理範圍內，但線性模型對於極端的組合可能會產生不切實際的預測。")
