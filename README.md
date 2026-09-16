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

## Task 7: cQuant Model-Ready Format

**Function:** `formatted_spot_history(df)`

- Splits data by settlement point.
- Pivots from one row per hour to one row per day, with 24 hourly price columns.
- Maps hour-beginning timestamps to columns: `00:00` is `X1`, `23:00` is `X24`.
- Adds a `Variable` column with the settlement point name.
- Writes 15 files to `output/formattedSpotHistory/spot_<SettlementPoint>.csv`.

## Bonus: Monthly Mean Price Plots

**Function:** `mean_plots(avg)`

- Assigns each monthly average to the first day of its month to create a chronological date axis.
- Plots one line per settlement point, with a legend identifying each curve.
- Writes `output/SettlementHubAveragePriceByMonth.png` (hubs only) and `output/LoadZoneAveragePriceByMonth.png` (load zones only).

## Bonus: Volatility Plot

**Function:** `volatility_plot(vol)`

- Pivots Task 4 results so each year is a group and each hub is a bar.
- Writes `output/HourlyVolatilityByHubAndYear.png`.

## Bonus: Hourly Shape Profiles

**Function:** `hourly_shape_profiles(df)`

- Averages hourly prices by month of year, day of week (0 = Monday), and hour of day.
- Divides each 24-hour profile by its own mean so every profile averages exactly 1.
- Writes 84 profiles (12 months x 7 days) per settlement point to `output/hourlyShapeProfiles/profile_<SettlementPoint>.csv`.
- HB_PAN only has data for April to December 2019, so its January to March profiles are blank.