# cQuant / Zema Global Energy Analyst Programming Exercise

## Task 1: Combine the Data

**Function:** `load_data()`

- Reads every CSV in `historicalPriceData` and stacks them with `pd.concat`.
- Parses `Date` into datetimes and adds `Year`, `Month`, `Hour`, and `DayOfWeek`.
- Sorts by settlement point, then time.

## Task 2: Monthly Average Price

**Function:** `monthly_average(df)`

- Groups by settlement point, year, and month, then averages price.
- Keeps prices at or below zero.

## Task 3: Write Monthly Averages

**Function:** `write_monthly_average(avg)`

- Writes `AveragePriceByMonth.csv` with columns `SettlementPoint, Year, Month, AveragePrice`.

## Task 4: Hourly Volatility

**Function:** `hourly_volatility(df)`

1. Keeps hubs only (`HB_`) with prices greater than zero.
2. Sorts each hub's prices in time order.
3. Computes log returns, `ln(P_t) - ln(P_t-1)`, within each hub and year.
4. Takes the standard deviation of log returns per hub and year.

## Task 5: Write Volatilities

**Function:** `write_volatility(vol)`

- Writes `HourlyVolatilityByYear.csv` with columns `SettlementPoint, Year, HourlyVolatility`.

## Task 6: Most Volatile Hub per Year

**Function:** `max_volatility(vol)`

- Uses `groupby("Year").idxmax()` to select the highest-volatility row in each year.
- Writes `MaxVolatilityByYear.csv` with the same columns as Task 5.

## Task 7: cQuant