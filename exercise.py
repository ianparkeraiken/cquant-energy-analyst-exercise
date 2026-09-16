from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

##################################################################################
#Bonus 1
def mean_plots(avg: pd.DataFrame) -> None:
    """Monthly average price lines: one plot for hubs, one for load zones."""
    avg = avg.assign(
        MonthStart=pd.to_datetime(dict(year=avg["Year"], month=avg["Month"], day=1))
    )
    plots = [
        ("HB_", "Monthly Average Price by Settlement Hub", "SettlementHubAveragePriceByMonth.png"),
        ("LZ_", "Monthly Average Price by Load Zone", "LoadZoneAveragePriceByMonth.png"),
    ]
    for prefix, title, filename in plots:
        fig, ax = plt.subplots(figsize=(14, 6))
        subset = avg[avg["SettlementPoint"].str.startswith(prefix)]
        for sp, g in subset.groupby("SettlementPoint"):
            g = g.sort_values("MonthStart")
            ax.plot(g["MonthStart"], g["AveragePrice"], label=sp)
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Price ($/MWh)")
        ax.legend(title="Settlement Point", loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=150)
        plt.close(fig)
        print(f"Bonus: wrote {filename}")

#Bonus 2
def volatility_plot(vol: pd.DataFrame) -> None:
    """Grouped bar chart comparing hub volatility by year."""
    wide = vol.pivot(index="Year", columns="SettlementPoint", values="HourlyVolatility")
    ax = wide.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title("Hourly Price Volatility by Settlement Hub and Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Hourly Volatility (std. dev. of log returns)")
    ax.legend(title="Settlement Hub", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "HourlyVolatilityByHubAndYear.png", dpi=150)
    plt.close()
    print("Bonus: wrote HourlyVolatilityByHubAndYear.png")

#Bonus 3
def hourly_shape_profiles(df: pd.DataFrame) -> None:
    """84 normalized 24-hour profiles (month x day of week) per settlement point."""
    prof_dir = OUT_DIR / "hourlyShapeProfiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    x_cols = [f"X{h + 1}" for h in range(24)]
    full_grid = pd.MultiIndex.from_product(
        [range(1, 13), range(7)], names=["Month", "DayOfWeek"]
    )

    for sp, g in df.groupby("SettlementPoint"):
        shape = (
            g.groupby(["Month", "DayOfWeek", "Hour"])["Price"].mean()
            .unstack("Hour")
            .reindex(columns=range(24))
        )
        row_mean = shape.mean(axis=1)
        if (row_mean <= 0).any():
            print(f"Bonus: WARNING {sp} has profiles with average price <= 0")
        shape = shape.div(row_mean, axis=0)   # each row now averages exactly 1
        shape = shape.reindex(full_grid)      # always 84 rows
        shape.columns = x_cols

        out = shape.reset_index()
        out.insert(0, "Variable", sp)
        filled = out.dropna(subset=x_cols, how="all")
        assert np.allclose(filled[x_cols].mean(axis=1), 1.0), f"{sp} normalization failed"
        out.to_csv(prof_dir / f"profile_{sp}.csv", index=False)
        print(f"Bonus: {sp}: {len(filled)} of 84 profiles filled")

    print("Bonus: profile files written =", len(list(prof_dir.glob("profile_*.csv"))))

#Bonus visualization of 24 hour profiles
def shape_profile_plot(sp: str = "HB_NORTH") -> None:
    """Plot selected normalized hourly shape profiles for one settlement point."""
    p = pd.read_csv(OUT_DIR / "hourlyShapeProfiles" / f"profile_{sp}.csv")
    x_cols = [f"X{h + 1}" for h in range(24)]
    hours = range(1, 25)

    def get_profile(month, dow):
        return p[(p["Month"] == month) & (p["DayOfWeek"] == dow)][x_cols].iloc[0].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Seasonal comparison, Wednesday (DayOfWeek 2)
    for month, name in [(1, "January"), (4, "April"), (7, "July"), (10, "October")]:
        ax1.plot(hours, get_profile(month, 2), marker="o", label=name)
    ax1.set_title(f"{sp} Wednesday Shape Profile by Season")

    # Weekday vs weekend, July
    ax2.plot(hours, get_profile(7, 2), marker="o", label="Wednesday")
    ax2.plot(hours, get_profile(7, 6), marker="o", label="Sunday")
    ax2.set_title(f"{sp} July Shape Profile: Weekday vs Weekend")

    for ax in (ax1, ax2):
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Hour Ending (X1 to X24)")
        ax.set_xticks(hours)
        ax.grid(alpha=0.3)
        ax.legend()
    ax1.set_ylabel("Normalized Price (daily average = 1)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"ShapeProfiles_{sp}.png", dpi=150)
    plt.close(fig)
    print(f"Bonus: wrote ShapeProfiles_{sp}.png")


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

    ##############################
    #Bonus 1
    mean_plots(avg)

    #Bonus 2
    volatility_plot(vol)

    #Bonus 3
    hourly_shape_profiles(df)

    #Bonus visualization of 24 hour profiles
    shape_profile_plot("HB_NORTH")