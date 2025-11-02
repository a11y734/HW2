# 🚗 車價預測專案 (Car Price Prediction)

本專案使用 Kaggle 上的[汽車價格資料集](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)，透過多元線性迴歸模型進行分析與價格預測。

專案利用 Streamlit 建立了一個互動式的 Web App，讓使用者可以：
- 探索單一汽車特徵與價格之間的關係。
- 了解模型自動篩選出的重要特徵及其對價格的影響。
- 透過調整各項參數，即時獲得車價預測。

## ✨ 技術棧

- **資料分析與模型訓練**: Pandas, Scikit-learn, Statsmodels
- **Web App**: Streamlit
- **資料視覺化**: Matplotlib, Seaborn

## 📊 專案流程 (CRISP-DM)

本專案遵循 CRISP-DM (Cross-Industry Standard Process for Data Mining) 流程進行開發：

1.  **商業理解 (Business Understanding)**
    *   **目標**: 建立一個能夠根據汽車的各項規格（如引擎大小、馬力、車重等）來預測其市場價格的模型。
    *   **效益**: 讓使用者能透過互動介面，直觀地了解不同配備對車價的影響，並獲得一個合理的價格預估。

2.  **資料理解 (Data Understanding)**
    *   **來源**: 使用 Kaggle 的 `CarPrice_Assignment.csv` 資料集。
    *   **探索**: 透過 Streamlit 的「單一特徵迴歸分析」功能，視覺化探索單一變數與目標變數 `price` 之間的關係。

3.  **資料準備 (Data Preparation)**
    *   在 `load_data` 函數中，移除了不必要的欄位 (`car_ID`, `CarName`)。
    *   將所有類別型特徵（如 `carbody`, `enginetype`）轉換為模型可以處理的數值格式（One-Hot Encoding）。

4.  **模型建立 (Modeling)**
    *   在 `train_model` 函數中，使用 `sklearn.linear_model.LinearRegression` 作為基礎模型。
    *   透過遞歸特徵消除 (RFE) 技術，從眾多特徵中自動篩選出 15 個最具預測能力的特徵組合。

5.  **模型評估 (Evaluation)**
    *   使用 R² 分數和均方根誤差 (RMSE) 來評估模型在測試集上的表現。
    *   透過「整體模型預測效果」圖表，視覺化比較預測價格與實際價格的差異，並提供 95% 預測區間。

6.  **部署 (Deployment)**
    *   使用 Streamlit 將整個分析流程與預測模型打包成一個互動式的 Web App (`main.py`)，讓非技術背景的使用者也能輕鬆操作。

## � 如何執行

1. **安裝依賴套件**:
   ```bash
   pip install -r requirements.txt
   ```
2. **啟動 Streamlit App**:
   ```bash
   streamlit run main.py
   ```