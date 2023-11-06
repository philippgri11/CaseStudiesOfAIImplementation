library(forecast)
library(ggplot2)
library(readxl)
library(dplyr)
library(tidyr)
library(lubridate)

# Read the specific sheet from the Excel file
LoadCurve2data <- read_excel("E:/OVGU university/semester3-2023.24/AI/training_data_period_1.xlsx", sheet = "LoadCurve2")

# Assuming the first column is the date and the third column is the load
LoadCurve2_df <- data.frame(DateTime = LoadCurve2data[[1]], Load = LoadCurve2data[[3]])

# Convert the DateTime to Date class and extract Year, Month, Day, Hour, and Week
LoadCurve2_df <- LoadCurve2_df %>%
  mutate(Date = as.Date(DateTime),
         Year = year(Date),
         Month = month(Date),
         Day = day(Date),
         Hour = hour(DateTime),
         Week = week(Date),
         Season = ifelse(Month %in% c(12, 1, 2), 'Winter',
                         ifelse(Month %in% c(3, 4, 5), 'Spring',
                                ifelse(Month %in% c(6, 7, 8), 'Summer', 'Fall')))) %>%
  select(-DateTime) # Remove the original DateTime column if it's no longer needed

# Define the seasons in order
seasons_ordered <- c("Winter", "Spring", "Summer", "Fall")

# Convert 'Season' to a factor with levels in the correct order
LoadCurve2_df$Season <- factor(LoadCurve2_df$Season, levels = seasons_ordered)


# Remove rows with NA values in the 'Load' column
LoadCurve2_df <- LoadCurve2_df %>% 
  drop_na(Load)


# Aggregate data by day
daily_aggregate <- LoadCurve2_df %>%
  group_by(Year, Month, Day) %>%
  summarise(DailyLoad = mean(Load))

# Plot daily seasonality
ggplot(daily_aggregate, aes(x = as.Date(paste(Year, Month, Day, sep="-")), y = DailyLoad)) +
  geom_line() +
  ggtitle("Daily Load Trend") +
  xlab("Date") +
  ylab("Average Load")

# For the functions from the forecast package, convert the data to a ts object
daily_ts <- ts(daily_aggregate$DailyLoad, frequency = 365)

# Use autoplot to plot a ts object
autoplot(daily_ts)

# Seasonal plots with the forecast package
ggseasonplot(daily_ts)
ggseasonplot(daily_ts, polar=TRUE)

# Histogram of daily loads
hist(daily_aggregate$DailyLoad)

# Subseries plot
ggsubseriesplot(daily_ts)

# Autocorrelation function plot
ggAcf(daily_ts)
ggPacf(daily_ts)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Aggregate data by hour
hourly_aggregate <- LoadCurve2_df %>%
  group_by(Year, Day, Hour) %>%
  summarise(HourlyLoad = mean(Load, na.rm = TRUE), .groups = 'drop')  # Use na.rm = TRUE to remove NA values

# Plot hourly seasonality
ggplot(hourly_aggregate, aes(x = Hour, y = HourlyLoad, group = Day)) +
  geom_line() +
  facet_wrap(~Year) +
  ggtitle("Hourly Load Trend") +
  xlab("Hour of the Day") +
  ylab("Average Load")


# Convert to a ts object with frequency equal to the number of hours in a day
hourly_ts <- ts(hourly_aggregate$HourlyLoad, frequency = 24)

# Now you can use the forecast package plotting functions
autoplot(hourly_ts)
ggseasonplot(hourly_ts)
ggseasonplot(hourly_ts, polar=TRUE)
hist(hourly_ts)
ggsubseriesplot(hourly_ts)
ggAcf(hourly_ts)
ggPacf(hourly_ts)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Aggregate data by week
weekly_aggregate <- LoadCurve2_df %>%
  group_by(Year, Week) %>%
  summarise(WeeklyLoad = mean(Load))

# Plot weekly seasonality
ggplot(weekly_aggregate, aes(x = Week, y = WeeklyLoad, group = Year)) +
  geom_line() +
  facet_wrap(~Year) +
  ggtitle("Weekly Load Trend") +
  xlab("Week of the Year") +
  ylab("Average Load")

weekly_ts <- ts(weekly_aggregate$WeeklyLoad, frequency = 52)

# Now you can use the forecast package plotting functions
autoplot(weekly_ts)
ggseasonplot(weekly_ts)
ggseasonplot(weekly_ts, polar=TRUE)
hist(weekly_ts)
ggsubseriesplot(weekly_ts)
ggAcf(weekly_ts)
ggPacf(weekly_ts)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Aggregate data by month
monthly_aggregate <- LoadCurve2_df %>%
  group_by(Year, Month) %>%
  summarise(MonthlyLoad = mean(Load))

# Plot monthly seasonality
ggplot(monthly_aggregate, aes(x = Month, y = MonthlyLoad, group = Year)) +
  geom_line() +
  facet_wrap(~Year) +
  ggtitle("Monthly Load Trend") +
  xlab("Month") +
  ylab("Average Load")

monthly_ts <- ts(monthly_aggregate$MonthlyLoad, frequency = 12)
autoplot(monthly_ts)
ggseasonplot(monthly_ts)
ggseasonplot(monthly_ts, polar=TRUE)
hist(monthly_ts, main = "Histogram of Monthly Loads", xlab = "Load")
ggsubseriesplot(monthly_ts)
ggAcf(monthly_ts)
ggPacf(monthly_ts)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Aggregate data by season
seasonal_aggregate <- LoadCurve2_df %>%
  group_by(Year, Season) %>%
  summarise(SeasonalLoad = mean(Load, na.rm = TRUE), .groups = 'drop')  # Use na.rm = TRUE to remove NA values

# Plot seasonal trends
ggplot(seasonal_aggregate, aes(x = Season, y = SeasonalLoad, group = Year)) +
  geom_line() +
  facet_wrap(~Year) +
  ggtitle("Seasonal Load Trend") +
  xlab("Season") +
  ylab("Average Load")
seasonal_ts <- ts(seasonal_aggregate$SeasonalLoad, frequency = 4)
autoplot(seasonal_ts)
ggseasonplot(seasonal_ts)
ggseasonplot(seasonal_ts, polar=TRUE)
hist(seasonal_ts, main = "Histogram of seasonal Loads", xlab = "Load")
ggsubseriesplot(seasonal_ts)
ggAcf(seasonal_ts)
ggPacf(seasonal_ts)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Aggregate data by year
yearly_aggregate <- LoadCurve2_df %>%
  group_by(Year) %>%
  summarise(YearlyLoad = mean(Load))

# Plot the yearly trend
ggplot(yearly_aggregate, aes(x = Year, y = YearlyLoad)) +
  geom_line() +
  geom_point() +
  ggtitle("Yearly Load Trend") +
  xlab("Year") +
  ylab("Average Load")

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$







# Aggregate data by hour
hourly_aggregate <- LoadCurve2_df %>%
  group_by(Year,Hour) %>%
  summarise(HourlyLoad = mean(Load))

# Plot hourly seasonality
ggplot(hourly_aggregate, aes(x = Hour, y = HourlyLoad)) +
  geom_line() +
  facet_wrap(~Year) +
  ggtitle("Hourly Load Trend") +
  xlab("Hour of the Day") +
  ylab("Average Load")
