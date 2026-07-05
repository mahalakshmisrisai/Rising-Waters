"""
generate_dataset.py
--------------------
Generates a synthetic historical weather dataset for flood prediction.

Replace this with your real historical weather dataset (CSV) if you have one.
The CSV must contain these columns (feature names can be remapped in
train_model.py -> FEATURE_COLUMNS):

    annual_rainfall     -> total rainfall recorded in the year (mm)
    seasonal_rainfall    -> rainfall recorded during the monsoon/rainy season (mm)
    cloud_visibility     -> visibility affected by cloud cover (km, lower = denser cloud)
    temperature          -> average temperature (Celsius)
    humidity             -> average relative humidity (%)
    flood                -> target label, 1 = flood occurred, 0 = no flood
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 3000

# --- Generate base meteorological features -------------------------------
annual_rainfall = np.random.normal(1800, 500, N_SAMPLES).clip(200, 4500)
seasonal_rainfall = annual_rainfall * np.random.uniform(0.45, 0.75, N_SAMPLES)
cloud_visibility = np.random.normal(6, 2.5, N_SAMPLES).clip(0.2, 15)   # km
temperature = np.random.normal(28, 4, N_SAMPLES).clip(15, 42)          # Celsius
humidity = np.random.normal(70, 12, N_SAMPLES).clip(30, 100)           # %

# --- Construct a flood "risk score" from realistic relationships ---------
# High seasonal rainfall + low cloud visibility (thick cloud/heavy rain) +
# high humidity increase flood risk.
risk_score = (
    0.55 * (seasonal_rainfall / seasonal_rainfall.max())
    + 0.25 * (1 - cloud_visibility / cloud_visibility.max())
    + 0.15 * (humidity / humidity.max())
    + 0.05 * (annual_rainfall / annual_rainfall.max())
)

# Add noise so the problem isn't trivially separable
risk_score += np.random.normal(0, 0.06, N_SAMPLES)

threshold = np.percentile(risk_score, 65)  # ~35% flood-positive rate
flood = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "annual_rainfall": annual_rainfall.round(2),
    "seasonal_rainfall": seasonal_rainfall.round(2),
    "cloud_visibility": cloud_visibility.round(2),
    "temperature": temperature.round(2),
    "humidity": humidity.round(2),
    "flood": flood,
})

df.to_csv("flood_data.csv", index=False)
print(f"Generated flood_data.csv with {len(df)} rows.")
print(df["flood"].value_counts(normalize=True).rename("proportion"))
