"""Generate synthetic hourly port activity data.

All numbers are simulated. This is NOT real port data.
Run from the project root:  python data/generate_synthetic_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DAYS = 120
CONGESTION_QUEUE_THRESHOLD = 25  # trucks waiting at the gate


def generate(days: int = DAYS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    df = pd.DataFrame({"timestamp": hours})
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Trucks arrive more during working hours and on weekdays.
    daytime_boost = np.where((df["hour"] >= 7) & (df["hour"] <= 18), 25, 5)
    weekday_boost = np.where(df["day_of_week"] < 5, 10, -5)
    df["scheduled_trucks"] = np.clip(
        rng.normal(daytime_boost + weekday_boost, 6), 0, None
    ).round().astype(int)

    df["vessels_in_port"] = rng.integers(0, 6, size=len(df))
    df["customs_delay_min"] = np.clip(rng.normal(30, 15, len(df)), 5, None).round(1)

    # Queue grows with trucks, vessels and customs delay, plus noise.
    queue = (
        0.55 * df["scheduled_trucks"]
        + 2.2 * df["vessels_in_port"]
        + 0.12 * df["customs_delay_min"]
        + rng.normal(0, 3, len(df))
    )
    df["queue_length"] = np.clip(queue, 0, None).round().astype(int)
    df["congested"] = (df["queue_length"] >= CONGESTION_QUEUE_THRESHOLD).astype(int)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "port_data.csv"
    data = generate()
    data.to_csv(out, index=False)
    print(f"Saved {len(data)} rows to {out}")
    print(f"Share of congested hours: {data['congested'].mean():.1%}")
