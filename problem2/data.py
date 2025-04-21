import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Load the Excel file
def load_data() -> pd.DataFrame:

    # 1) Load and reshape
    file_path = "C:/Users/AJAX/PycharmProjects/QuantChallenge/problem2/analysis_task_data.xlsx"
    df = pd.read_excel(file_path, sheet_name=0)

    # check for duplicate in th
    df = df.drop_duplicates(subset=['time'])

    # set timestamp as the index
    df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df.set_index('timestamp', inplace=True)

    df.drop(columns=['time', 'hour'], errors='ignore', inplace=True)

    # fill in missing data
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].interpolate(method='time')
    df[num_cols] = df[num_cols].bfill().ffill()

    return df


def generate_graphs(df: pd.DataFrame):
    # Set the theme for time series aesthetics
    sns.set_theme(style="darkgrid", context="notebook", palette="muted")

    # Ensure time index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)

    # Wind Forecasts
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=df[['Wind Day Ahead Forecast [in MW]', 'Wind Intraday Forecast [in MW]']])
    plt.title('Wind Power Forecasts (MW)', fontsize=20)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Forecasted Wind Power [MW]', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # PV Forecasts
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=df[['PV Day Ahead Forecast [in MW]', 'PV Intraday Forecast [in MW]']])
    plt.title('Solar PV Power Forecasts (MW)', fontsize=20)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Forecasted PV Power [MW]', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Prices
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=df[['Day Ahead Price hourly [in EUR/MWh]',
                          'Intraday Price Price Quarter Hourly  [in EUR/MWh]']])
    plt.title('Power Market Prices (EUR/MWh)', fontsize=20)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price [EUR/MWh]', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_result_problem_2_1(df: pd.DataFrame()):
    # 2.1 Total forecasted energy in 2021 (MWh)
    df['Wind_DA_MWh'] = df['Wind Day Ahead Forecast [in MW]'] * 0.25
    df['Wind_ID_MWh'] = df['Wind Intraday Forecast [in MW]'] * 0.25
    df['PV_DA_MWh'] = df['PV Day Ahead Forecast [in MW]'] * 0.25
    df['PV_ID_MWh'] = df['PV Intraday Forecast [in MW]'] * 0.25
    totals = {
        'Wind DA (MWh)': df['Wind_DA_MWh'].sum(),
        'Wind ID (MWh)': df['Wind_ID_MWh'].sum(),
        'PV DA (MWh)': df['PV_DA_MWh'].sum(),
        'PV ID (MWh)': df['PV_ID_MWh'].sum(),
    }
    print("Task 2.1 totals:")
    for key, val in totals.items():
        print(f" {key}: {val:,.0f} MWh")

    print("\n")

    # Plot pie chart
    total_sum = sum(totals.values())
    labels = [f"{k}\n{v:,.0f} MWh" for k, v in totals.items()]
    plt.figure(figsize=(8, 6))
    plt.pie(totals.values(), labels=labels, autopct=lambda p: f'{p:.1f}%\n({p * total_sum / 100:,.0f} MWh)',
            startangle=90, colors=sns.color_palette("pastel"))
    plt.title(f"Total Annual Forecasted Energy (MWh)\nTotal = {total_sum:,.0f} MWh")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def generate_result_problem_2_2(df: pd.DataFrame()):
    # 2.2 Average daily profile by hour
    hourly_avg = df.groupby(df.index.hour)[[
        'Wind Day Ahead Forecast [in MW]',
        'Wind Intraday Forecast [in MW]',
        'PV Day Ahead Forecast [in MW]',
        'PV Intraday Forecast [in MW]'
    ]].mean()

    plt.figure(figsize=(12, 6))
    for col in hourly_avg.columns:
        plt.plot(hourly_avg.index, hourly_avg[col], label=col)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Task 2.2: Average Daily Profile by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Forecast (MW)")
    plt.xticks(range(0, 24))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def generate_result_problem_2_3(df: pd.DataFrame()):
    # 2.3 Weighted average value at DA prices
    df['prod_Wind_DA_MWh'] = df['Wind Day Ahead Forecast [in MW]'] * 0.25
    df['prod_PV_DA_MWh'] = df['PV Day Ahead Forecast [in MW]'] * 0.25
    da_price = df['Day Ahead Price hourly [in EUR/MWh]']

    avg_wind_val = (da_price * df['prod_Wind_DA_MWh']).sum() / df['prod_Wind_DA_MWh'].sum()
    avg_pv_val = (da_price * df['prod_PV_DA_MWh']).sum() / df['prod_PV_DA_MWh'].sum()
    print(
        f"Task 2.3:\n Wind owner avg = {avg_wind_val:.2f} €/MWh\n PV owner avg  = {avg_pv_val:.2f} €/MWh\n Overall DA avg price = {da_price.mean():.2f} €/MWh\n")


def generate_result_problem_2_4(df: pd.DataFrame()):
    # 2.4 High/low renewable days
    df['date'] = df.index.date
    daily_prod = df.groupby('date')[['Wind_DA_MWh', 'PV_DA_MWh']].sum()
    daily_prod['total_MWh'] = daily_prod.sum(axis=1)

    max_day = daily_prod['total_MWh'].idxmax()
    min_day = daily_prod['total_MWh'].idxmin()

    hourly = df.resample('h').first()
    daily_da_price = hourly.groupby(hourly.index.date)['Day Ahead Price hourly [in EUR/MWh]'].mean()

    print("Task 2.4:")
    print(
        f" Highest renewables {max_day}: {daily_prod.loc[max_day, 'total_MWh']:.0f} MWh, avg DA price {daily_da_price.loc[max_day]:.2f} €/MWh")
    print(
        f" Lowest  renewables {min_day}: {daily_prod.loc[min_day, 'total_MWh']:.0f} MWh, avg DA price {daily_da_price.loc[min_day]:.2f} €/MWh\n")

    plt.figure(figsize=(14, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(daily_prod.index, daily_prod['total_MWh'], color='green', label='Renewable Production (MWh)', linewidth=2)
    ax2.plot(daily_da_price.index, daily_da_price, color='blue', label='Avg DA Price (EUR/MWh)', linewidth=2)

    ax1.scatter([max_day], [daily_prod.loc[max_day, 'total_MWh']], color='darkgreen', s=80, zorder=5,
                label='Max Renewable')
    ax1.scatter([min_day], [daily_prod.loc[min_day, 'total_MWh']], color='red', s=80, zorder=5, label='Min Renewable')

    ax1.set_ylabel('Renewable Production (MWh)', color='green')
    ax2.set_ylabel('Avg DA Price (EUR/MWh)', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_title('Daily Renewable Production vs Day-Ahead Price in 2021')
    ax1.grid(True, linestyle='--', alpha=0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()


def generate_result_problem_2_5(df: pd.DataFrame):
    # 2.5 Avg DA price weekday vs weekend
    hourly = df.resample('h').first()

    hourly['is_weekday'] = hourly.index.dayofweek < 5
    hourly['hour'] = hourly.index.hour

    wd_avg = hourly[hourly['is_weekday']]['Day Ahead Price hourly [in EUR/MWh]'].mean()
    we_avg = hourly[~hourly['is_weekday']]['Day Ahead Price hourly [in EUR/MWh]'].mean()
    print(f"Task 2.5: Weekday avg = {wd_avg:.2f} €/MWh, Weekend avg = {we_avg:.2f} €/MWh\n")

    weekday_avg_price = hourly[hourly['is_weekday']].groupby('hour')['Day Ahead Price hourly [in EUR/MWh]'].mean()
    weekend_avg_price = hourly[~hourly['is_weekday']].groupby('hour')['Day Ahead Price hourly [in EUR/MWh]'].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(weekday_avg_price.index, weekday_avg_price, label='Weekday Avg Price', color='blue', linewidth=2)
    plt.plot(weekend_avg_price.index, weekend_avg_price, label='Weekend Avg Price', color='orange', linewidth=2)

    plt.axhline(wd_avg, color='blue', linestyle='--', linewidth=1.5, label=f'Weekday Mean ({wd_avg:.2f} €/MWh)')
    plt.axhline(we_avg, color='orange', linestyle='--', linewidth=1.5, label=f'Weekend Mean ({we_avg:.2f} €/MWh)')

    plt.title('Average Hourly Day-Ahead Price: Weekdays vs Weekends')
    plt.xlabel('Hour of Day')
    plt.ylabel('Avg DA Price (EUR/MWh)')
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_result_problem_2_6a(df: pd.DataFrame) -> pd.DataFrame:
    # Resample to hourly data
    hourly = df.resample('h').first()
    prices = hourly['Day Ahead Price hourly [in EUR/MWh]']
    records = []

    BATTERY_CAPACITY_MWH = 1.0  # 1 MWh capacity

    for date, group in prices.groupby(prices.index.date):
        if group.isnull().any() or len(group) < 2:
            continue

        # Initialize variables for tracking the best revenue
        best_revenue = 0.0
        best_charge_price = None
        best_charge_time = None
        best_discharge_price = None
        best_discharge_time = None

        # Convert to numpy arrays for faster processing
        price_series = group.values
        time_series = group.index

        # We need to find the minimum price and maximum price where max time > min time
        for i in range(len(price_series)):
            min_price = price_series[i]
            min_time = time_series[i]

            # Find the maximum price after the minimum price (max price time must be after min price time)
            for j in range(i + 1, len(price_series)):
                max_price = price_series[j]
                max_time = time_series[j]

                # If max price occurs after min price time, calculate revenue
                if max_price > min_price:
                    revenue = (max_price - min_price) * BATTERY_CAPACITY_MWH

                    # Track the best (maximum) revenue and corresponding times and prices
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_charge_price = min_price
                        best_charge_time = min_time
                        best_discharge_price = max_price
                        best_discharge_time = max_time

        # If we found a valid charge-discharge cycle, record it
        if best_revenue > 0:
            records.append({'date': pd.to_datetime(date), 'revenue': best_revenue,
                            'charge': best_charge_time, 'discharge': best_discharge_time,
                            'charge_price': best_charge_price, 'discharge_price': best_discharge_price})

    # Create DataFrame with the results
    revenue_df = pd.DataFrame(records).set_index('date')
    total_revenue = revenue_df['revenue'].sum()
    print(f"Task 2.6a (Best Revenue for single cycle per day): Total 2021 battery revenue = €{total_revenue:.2f}\n")

    # Plotting the result: Revenue over time and Charge/Discharge prices
    plt.figure(figsize=(12, 6))

    # Plot revenue over time
    plt.subplot(2, 1, 1)
    plt.plot(revenue_df.index, revenue_df['revenue'], marker='o', color='b', label='Revenue (€)')
    plt.title('Battery Revenue Over Time')
    plt.xlabel('Date')
    plt.ylabel('Revenue (€)')
    plt.grid(True)
    plt.legend()

    # Plot charge and discharge prices over time
    plt.subplot(2, 1, 2)
    plt.plot(revenue_df.index, revenue_df['charge_price'], marker='o', color='g', label='Charge Price (€)')
    plt.plot(revenue_df.index, revenue_df['discharge_price'], marker='o', color='r', label='Discharge Price (€)')
    plt.title('Charge and Discharge Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (€)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return revenue_df


def generate_result_problem_2_6b(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.resample('h').first()
    prices = hourly['Day Ahead Price hourly [in EUR/MWh]']
    records = []

    BATTERY_CAPACITY_MWH = 1.0  # 1 MWh capacity

    for date, group in prices.groupby(prices.index.date):
        if group.isnull().any() or len(group) < 2:
            continue

        price_series = group.values

        i = 0
        revenue = 0.0
        cycles = 0

        while i < len(price_series) - 1:
            # Find local minimum (charge)
            while i < len(price_series) - 1 and price_series[i + 1] <= price_series[i]:
                i += 1
            charge_price = price_series[i]

            # Move to the next point
            i += 1
            if i >= len(price_series):
                break

            # Find local maximum (discharge)
            while i < len(price_series) - 1 and price_series[i + 1] >= price_series[i]:
                i += 1
            discharge_price = price_series[i]

            # Only count if there's positive revenue
            if discharge_price > charge_price:
                revenue += (discharge_price - charge_price) * BATTERY_CAPACITY_MWH
                cycles += 1

        if cycles > 0:
            records.append({
                'date': pd.to_datetime(date),
                'revenue': revenue,
                'num_cycles': cycles
            })

    result_df = pd.DataFrame(records).set_index('date')
    total_revenue = result_df['revenue'].sum()
    total_cycles = result_df['num_cycles'].sum()
    print(f"Task 2.6b (Greedy Multi-Cycle Strategy): Total 2021 battery revenue = €{total_revenue:.2f} across {total_cycles} cycles\n")

    # Plotting total daily revenue and number of cycles
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of Cycles', color=color)
    ax2.plot(result_df.index, result_df['num_cycles'], color=color, alpha=0.5, marker='x', label='Cycles')
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Revenue (€)', color=color)
    ax1.plot(result_df.index, result_df['revenue'], color=color, marker='o', label='Revenue')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    plt.title('Battery Revenue and Charge/Discharge Cycles per Day (Greedy Strategy)')
    fig.tight_layout()
    plt.show()

    return result_df


def generate_result_problem_2_7(
        df: pd.DataFrame,
        alpha: float ,
        up_quantile: float,
        down_quantile: float,
        position_mw: float = 100,
        season_filter: str = None,
        hour_filter: list = None,
        is_weekend: bool = None
) -> None:
    """
    Renewable-surprise arbitrage strategy between day-ahead and intraday hourly prices.
    Computes signals from forecast differences, trades based on quantiles,
    calculates detailed PnL and performance metrics, and plots results.
    """

    # Rename columns using exact sheet headers
    cols = {
        'Wind Day Ahead Forecast [in MW]': 'wind_DA',
        'Wind Intraday Forecast [in MW]': 'wind_ID',
        'PV Day Ahead Forecast [in MW]': 'pv_DA',
        'PV Intraday Forecast [in MW]': 'pv_ID',
        'Day Ahead Price hourly [in EUR/MWh]': 'DA_price',
        'Intraday Price Hourly  [in EUR/MWh]': 'ID_price'
    }
    df = df.rename(columns=cols)

    # Resample to hourly data
    hourly = df[list(cols.values())].resample('h').first().dropna()
    hourly = hourly[(hourly['wind_DA'] != 0) & (hourly['pv_DA'] != 0)]

    # Add filters
    # Season column
    hourly['season'] = hourly.index.month % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall

    # Hour of day column
    hourly['hour_of_day'] = hourly.index.hour

    # Weekday/weekend column (True if weekend, False if weekday)
    hourly['is_weekend'] = hourly.index.weekday >= 5  # Saturday (5) or Sunday (6)

    if season_filter:
        seasonal_map = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
        if season_filter in seasonal_map:
            hourly = hourly[hourly['season'] == seasonal_map[season_filter]]

    if hour_filter:
        hourly = hourly[hourly['hour_of_day'].isin(hour_filter)]

    if is_weekend is not None:
        hourly = hourly[hourly['is_weekend'] == is_weekend]

    # Signal logic
    hourly['S_wind'] = (hourly['wind_ID'] - hourly['wind_DA']) / hourly['wind_DA']
    hourly['S_pv'] = (hourly['pv_ID'] - hourly['pv_DA']) / hourly['pv_DA']
    hourly['S'] = alpha * hourly['S_wind'] + (1 - alpha) * hourly['S_pv']

    # standardization signal scaling
    mean_s = hourly['S'].mean()
    std_s = hourly['S'].std()
    hourly['S_scaled'] = (hourly['S'] - mean_s) / std_s

    # Signal thresholds
    T_up = hourly['S_scaled'].quantile(up_quantile)
    T_down = hourly['S_scaled'].quantile(down_quantile)

    # Trading signals
    hourly['signal'] = 0
    hourly.loc[hourly['S_scaled'] > T_up, 'signal'] = -1  # short DA, buy ID
    hourly.loc[hourly['S_scaled'] < T_down, 'signal'] = 1  # buy DA, short ID

    # Position Management
    hourly['position'] = hourly['signal'].replace(to_replace=0, method='ffill').fillna(0)
    hourly['position_shifted'] = hourly['position'].shift(1).fillna(0)
    hourly['position_change'] = hourly['position'] != hourly['position_shifted']

    # Trade Construction
    trades = []
    entry_idx = None
    entry_price = None
    entry_time = None
    entry_signal = None

    for idx, row in hourly.iterrows():
        if row['position_change']:
            # Exit trade
            if entry_signal is not None and entry_signal != 0:
                exit_time = idx
                exit_price = row['ID_price']
                pnl_per_mwh = (exit_price - entry_price) * entry_signal
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'signal': entry_signal,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'trade_duration': exit_time - entry_time,
                    'pnl_per_mwh': pnl_per_mwh,
                    'pnl_mw': pnl_per_mwh * position_mw
                })
            # Start new trade
            if row['position'] != 0:
                entry_time = idx
                entry_price = row['DA_price']
                entry_signal = row['position']

    # Handle open trade at the end
    if entry_signal is not None and entry_signal != 0:
        exit_price = hourly.iloc[-1]['ID_price']
        exit_time = hourly.index[-1]
        pnl_per_mwh = (exit_price - entry_price) * entry_signal
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'signal': entry_signal,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_duration': exit_time - entry_time,
            'pnl_per_mwh': pnl_per_mwh,
            'pnl_mw': pnl_per_mwh * position_mw
        })

    # Create trade DataFrame
    trade_df = pd.DataFrame(trades)
    trade_df['cum_pnl'] = trade_df['pnl_mw'].cumsum()

    # Backfill cumulative PnL into hourly index
    hourly['cum_pnl'] = np.nan
    hourly.loc[trade_df['exit_time'], 'cum_pnl'] = trade_df['cum_pnl'].values
    hourly['cum_pnl'] = hourly['cum_pnl'].ffill().fillna(0)

    # Performance Metrics
    final_pnl = trade_df['pnl_mw'].sum()
    mean_pnl = trade_df['pnl_mw'].mean()
    std_pnl = trade_df['pnl_mw'].std()
    sharpe_ratio = mean_pnl / std_pnl if std_pnl != 0 else np.nan

    downside_returns = trade_df.loc[trade_df['pnl_mw'] < 0, 'pnl_mw']
    sortino_ratio = mean_pnl / downside_returns.std() if not downside_returns.empty else np.nan

    rolling_max = trade_df['cum_pnl'].cummax()
    drawdown = trade_df['cum_pnl'] - rolling_max
    max_drawdown = drawdown.min()

    risk_reward_ratio = abs(mean_pnl) / abs(max_drawdown) if max_drawdown != 0 else np.nan

    num_trades = len(trade_df)
    win_rate = (trade_df['pnl_mw'] > 0).mean()

    # Best and worst trades
    best_trade = trade_df.loc[trade_df['pnl_mw'].idxmax()] if not trade_df.empty else None
    worst_trade = trade_df.loc[trade_df['pnl_mw'].idxmin()] if not trade_df.empty else None

    # Summary Output
    print(f"\n Performance Summary:")
    print(f"----------------------------")
    print(f"Final Cumulative PnL   : {final_pnl:,.2f} EUR")
    print(f"Mean Trade PnL         : {mean_pnl:,.2f} EUR")
    print(f"Volatility (Trade)     : {std_pnl:,.2f} EUR")
    print(f"Sharpe Ratio           : {sharpe_ratio:.2f}")
    print(f"Sortino Ratio          : {sortino_ratio:.2f}")
    print(f"Risk-Reward Ratio      : {risk_reward_ratio:.2f}")
    print(f"Max Drawdown           : {max_drawdown:,.2f} EUR")
    print(f"Total Trades           : {num_trades}")
    print(f"Win Rate               : {win_rate:.2%}")

    if best_trade is not None:
        print(f"\nBest Trade:")
        print(f"Entry                  : {best_trade['entry_time']}")
        print(f"Exit                   : {best_trade['exit_time']}")
        print(f"Signal                 : {best_trade['signal']}")
        print(f"PnL                    : {best_trade['pnl_mw']:,.2f} EUR")

    if worst_trade is not None:
        print(f"\nWorst Trade:")
        print(f"Entry                  : {worst_trade['entry_time']}")
        print(f"Exit                   : {worst_trade['exit_time']}")
        print(f"Signal                 : {worst_trade['signal']}")
        print(f"PnL                    : {worst_trade['pnl_mw']:,.2f} EUR")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(hourly.index, hourly['cum_pnl'], label='Cumulative PnL', linewidth=2, color='blue')

        if best_trade is not None:
            plt.axvspan(best_trade['entry_time'], best_trade['exit_time'], color='green', alpha=0.2, label='Best Trade')
        if worst_trade is not None:
            plt.axvspan(worst_trade['entry_time'], worst_trade['exit_time'], color='red', alpha=0.2,
                        label='Worst Trade')

        # Highlight +1 (buy) signal regions in green
        buy_signals = hourly['signal'] == 1
        plt.fill_between(hourly.index, hourly['cum_pnl'], where=buy_signals, color='green', alpha=0.1,
                         label='Buy Signal (+1)')

        # Highlight -1 (sell) signal regions in red
        sell_signals = hourly['signal'] == -1
        plt.fill_between(hourly.index, hourly['cum_pnl'], where=sell_signals, color='red', alpha=0.1,
                         label='Sell Signal (-1)')

        plt.title('DA–ID Renewable Surprise Strategy Cumulative PnL (2021)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL (EUR)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Trade Log
        print("\n📜 Trade Log (Sample):")
        print(trade_df.head(10))

if __name__ == "__main__":
    df = load_data()

    generate_result_problem_2_1(df)
    generate_result_problem_2_2(df)
    generate_result_problem_2_3(df)
    generate_result_problem_2_4(df)
    generate_result_problem_2_5(df)
    generate_result_problem_2_6a(df)
    generate_result_problem_2_6b(df)

    # season_filter = ["Winter":(1,3), "Spring":(4,6), "Summer":(7,9), "Fall":(10,12)  ]
    # alpha -> weighting factor for Wind/PV ratio
    generate_result_problem_2_7(df,0.9, 0.75, 0.25,
                                 season_filter="Fall", hour_filter=[5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                                 is_weekend=True)
