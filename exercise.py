from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("historicalPriceData")
OUT_DIR = Path("output")

# TASK 1
def load_data() -> pd.DataFrame:
    """Read every CSV in historicalPriceData and combine into one long table."""
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR.resolve()}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Hour"] = df["Date"].dt.hour          # hour-beginning: 0-23
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0 = Monday

    df = df.sort_values(["SettlementPoint", "Date"]).reset_index(drop=True)
    return df

#TASK 2
def monthly_average(df: pd.DataFrame) -> pd.DataFrame:
    avg = (
        df.groupby(["SettlementPoint", "Year", "Month"], as_index=False)["Price"]
        .mean()
        .rename(columns={"Price": "AveragePrice"})
    )
    print("Task 2: months per point\n", avg.groupby("SettlementPoint").size())
    return avg

#TASK 3
def write_monthly_average(avg: pd.DataFrame) -> None:
    cols = ["SettlementPoint", "Year", "Month", "AveragePrice"]
    avg[cols].to_csv(OUT_DIR / "AveragePriceByMonth.csv", index=False)
    print(f"Task 3: wrote AveragePriceByMonth.csv ({len(avg)} rows)")

#TASK 4
def hourly_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Std of hourly log returns per hub and year. Hubs only, prices > 0."""
    is_hub = df["SettlementPoint"].str.startswith("HB_")
    hubs = df[is_hub & (df["Price"] > 0)].copy()
    print(f"Task 4: removed {(is_hub & (df['Price'] <= 0)).sum()} hub prices <= 0")

    hubs = hubs.sort_values(["SettlementPoint", "Date"])
    hubs["LogReturn"] = (
        np.log(hubs["Price"])
        .groupby([hubs["SettlementPoint"], hubs["Year"]])
        .diff()
    )

    vol = (
        hubs.groupby(["SettlementPoint", "Year"], as_index=False)["LogReturn"]
        .std()
        .rename(columns={"LogReturn": "HourlyVolatility"})
    )
    print("Task 4: hourly volatility\n", vol)
    return vol

#TASK 5
def write_volatility(vol: pd.DataFrame) -> None:
    cols = ["SettlementPoint", "Year", "HourlyVolatility"]
    vol[cols].to_csv(OUT_DIR / "HourlyVolatilityByYear.csv", index=False)
    print(f"Task 5: wrote HourlyVolatilityByYear.csv ({len(vol)} rows)")

#TASK 6
def max_volatility(vol: pd.DataFrame) -> pd.DataFrame:
    """Row with the highest hourly volatility for each year."""
    max_vol = (
        vol.loc[vol.groupby("Year")["HourlyVolatility"].idxmax()]
        .reset_index(drop=True)
    )
    cols = ["SettlementPoint", "Year", "HourlyVolatility"]
    max_vol[cols].to_csv(OUT_DIR / "MaxVolatilityByYear.csv", index=False)
    print("Task 6: max volatility by year\n", max_vol)
    return max_vol

#TASK 7
def formatted_spot_history(df: pd.DataFrame) -> None:
    """One wide CSV per settlement point: Variable, Date, X1..X24."""
    spot_dir = OUT_DIR / "formattedSpotHistory"
    spot_dir.mkdir(parents=True, exist_ok=True)
    x_cols = [f"X{h + 1}" for h in range(24)]  # hour-beginning 00:00 -> X1

    for sp, g in df.groupby("SettlementPoint"):
        wide = (
            g.assign(Day=g["Date"].dt.strftime("%Y-%m-%d"))
            .pivot(index="Day", columns="Hour", values="Price")
            .reindex(columns=range(24))
        )
        wide.columns = x_cols
        blanks = int(wide.isna().sum().sum())
        wide = wide.rename_axis("Date").reset_index()
        wide.insert(0, "Variable", sp)
        wide.to_csv(spot_dir / f"spot_{sp}.csv", index=False)
        print(f"Task 7: {sp}: {len(wide)} days, blank cells = {blanks}")

    print("Task 7: files written =", len(list(spot_dir.glob("spot_*.csv"))))

if __name__ == "__main__":
    #Task 1
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data()

    #Task 2
    avg = monthly_average(df)

    #Task 3
    write_monthly_average(avg)

    #Task 4
    vol = hourly_volatility(df)

    #Task 5
    write_volatility(vol)

    #Task 6
    max_vol = max_volatility(vol)

    #Task 7
    formatted_spot_history(df)
