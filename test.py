import pandas as pd
from pathlib import Path

# Step 1: 讀取 CSV 檔案
# 建立相對於目前 .py 檔案的檔案路徑，讓程式更穩健
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "CarPrice_Assignment.csv"
df = pd.read_csv(DATA_FILE)

# Step 2: 確認讀取成功
print("✅ 資料載入成功！")
print("資料筆數與欄位數：", df.shape)
print("前5筆資料：")
print(df.head())
