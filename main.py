from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from datetime import datetime, timezone

from problem1.data_analysis import compute_pnl_for_each_strategy, load_single_table_sqlite

# FastAPI app initialization
app = FastAPI(title="Energy Trading API", version="1.0.0")

#initiliaze dataframe
data_path = "C:/Users/AJAX/PycharmProjects/QuantChallenge/problem1/trades.sqlite"
df = load_single_table_sqlite(data_path)

# Define a PnL response model
class PnLResponse(BaseModel):
    strategy: str
    value: float
    unit: str = "euro"
    capture_time: str

# API endpoint to get PnL for a specific strategy
@app.get(
    "/v1/pnl/{strategy_id}",
    response_model=PnLResponse,
    summary="Returns the PnL of the corresponding strategy.",
    description="This endpoint computes and returns the profit and loss (PnL) of the given strategy. "
                "The PnL is calculated based on the sum of the incomes from each trade (buy/sell) "
                "for that particular strategy."
)
async def get_pnl(strategy_id: str):
    pnl_df = compute_pnl_for_each_strategy(df)
    strategy_pnl = pnl_df[pnl_df["strategy"] == strategy_id]

    if strategy_pnl.empty:
        raise HTTPException(status_code=404, detail="Strategy not found")

    pnl_value = strategy_pnl["pnl"].iloc[0]
    capture_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return PnLResponse(
        strategy=strategy_id,
        value=pnl_value,
        capture_time=capture_time
    )


# API endpoint to get PnL for all strategies
@app.get(
    "/v1/pnls",
    response_model=list[PnLResponse],
    summary="Returns the PnL for all strategies.",
    description="This endpoint computes and returns the profit and loss (PnL) for all strategies. "
                "The PnL for each strategy is calculated based on the sum of incomes from each trade "
                "(buy/sell) for all trades in the dataset."
)
async def get_all_pnls():
    pnl_df = compute_pnl_for_each_strategy(df)

    if pnl_df.empty:
        raise HTTPException(status_code=404, detail="No strategies found")

    capture_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return [
        PnLResponse(
            strategy=row["strategy"],
            value=row["pnl"],
            capture_time=capture_time
        ) for _, row in pnl_df.iterrows()
    ]
