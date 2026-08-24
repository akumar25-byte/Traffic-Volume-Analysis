import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv("traffic_volume_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Weekend"] = df["Day"].isin(["Saturday", "Sunday"])

print("First 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df["Traffic_Volume"].describe())

# 2. Average traffic by hour
hourly = df.groupby("Hour")["Traffic_Volume"].mean()
print("\nAverage traffic by hour:")
print(hourly)

peak_hour = hourly.idxmax()
print(f"\nPeak traffic hour: {peak_hour}:00")
print(f"Average traffic at peak hour: {hourly.max():.0f} vehicles")

# 3. Average traffic by day
daily = df.groupby("Date")["Traffic_Volume"].sum()
print("\nAverage daily traffic:", daily.mean())

# 4. Weekday vs weekend
day_type = df.groupby("Weekend")["Traffic_Volume"].mean()
print("\nWeekday vs weekend average hourly traffic:")
print(day_type.rename({False: "Weekday", True: "Weekend"}))

# 5. Vehicle composition
vehicle_cols = ["Cars", "Motorcycles", "Buses", "Trucks"]
vehicle_totals = df[vehicle_cols].sum()
print("\nVehicle composition:")
print(vehicle_totals)

# 6. Visualization: hourly traffic
plt.figure(figsize=(9, 5))
sns.lineplot(x=hourly.index, y=hourly.values, marker="o")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Vehicles")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Visualization: vehicle composition
plt.figure(figsize=(8, 5))
vehicle_totals.plot(kind="bar")
plt.title("Vehicle Type Composition")
plt.xlabel("Vehicle Type")
plt.ylabel("Total Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 8. Heatmap: date vs hour
heatmap_data = df.pivot_table(
    index="Date", columns="Hour", values="Traffic_Volume", aggfunc="mean"
)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="YlOrRd")
plt.title("Traffic Volume Heatmap")
plt.xlabel("Hour")
plt.ylabel("Date")
plt.tight_layout()
plt.show()

# 9. Weekday/weekend comparison
comparison = df.assign(
    Day_Type=df["Weekend"].map({False: "Weekday", True: "Weekend"})
).groupby(["Day_Type", "Hour"])["Traffic_Volume"].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=comparison, x="Hour", y="Traffic_Volume",
             hue="Day_Type", marker="o")
plt.title("Weekday vs Weekend Traffic")
plt.xlabel("Hour of Day")
plt.ylabel("Average Vehicles")
plt.tight_layout()
plt.show()
