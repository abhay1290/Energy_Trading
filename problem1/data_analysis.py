import sqlite3
import pandas as pd
from pathlib import Path

def load_single_table_sqlite(file_path: str) -> pd.DataFrame:
    db_path = Path(file_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    table_name = tables[0]

    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    return df

def compute_total_buy_volume(df: pd.DataFrame) -> float:
    return df[df["side"].str.lower() == "buy"]["quantity"].sum()

def compute_total_sell_volume(df: pd.DataFrame) -> float:
    return df[df["side"].str.lower() == "sell"]["quantity"].sum()

# PnL calculation function
def compute_pnl_for_each_strategy(df: pd.DataFrame) :
    if df.empty or "id" not in df.columns:
        return 0

    df["pnl"] = df["quantity"] * df["price"] * df["side"].apply(lambda side: 1 if side.lower() == "sell" else -1)
    pnl_by_strategy = df.groupby("strategy")["pnl"].sum().reset_index()
    return pnl_by_strategy


if __name__ == "__main__":
    data_path = "C:/Users/AJAX/PycharmProjects/QuantChallenge/problem1/trades.sqlite"
    df = load_single_table_sqlite(data_path)
    print(df)

    buy_volume = compute_total_buy_volume(df)
    sell_volume = compute_total_sell_volume(df)

    print(f"Total Buy Volume: {buy_volume}")
    print(f"Total Sell Volume: {sell_volume}")

    pnl_by_strategy = compute_pnl_for_each_strategy(df)

    print(f"PnL for each Strategy:\n{pnl_by_strategy}")



