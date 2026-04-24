import numpy as np
import pandas as pd

TARGET_LAGS = [1, 2, 4, 6, 8, 12, 24, 48]
TARGET_ROLLS = [3, 6, 12, 24, 48]
STATUS_ROLLS = [3, 6]
WH_ROLLS = [3, 6, 12, 24, 48]
GLOBAL_ROLLS = [3, 6, 12, 24, 48]
SHARE_LAGS = [1, 2, 4, 6, 12, 24]
FORECAST_POINTS = 10
TARGET_COL = "target_2h"

EXCLUDE_COLS = {
    TARGET_COL,
    "timestamp",
    "id",
    *[f"target_step_{s}" for s in range(1, FORECAST_POINTS + 1)],
}


def add_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    if target_col not in df.columns:
        df[target_col] = np.nan

    grp_route = df.groupby("route_id", sort=False)[target_col]

    for lag in TARGET_LAGS:
        df[f"target_lag_{lag}"] = grp_route.shift(lag)

    for w in TARGET_ROLLS:
        shifted = grp_route.shift(1)
        df[f"target_roll_mean_{w}"] = (
            shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"target_roll_std_{w}"] = (
            shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
        )
        df[f"target_roll_sum_{w}"] = (
            shifted.rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        )

    df["target_diff_1"] = grp_route.shift(1) - grp_route.shift(2)
    df["target_diff_2"] = grp_route.shift(1) - grp_route.shift(3)
    df["target_ratio_1_2"] = grp_route.shift(1) / (grp_route.shift(2) + 1e-6)

    status_cols = [c for c in df.columns if c.startswith("status_")]
    for col in status_cols:
        grp_s = df.groupby("route_id", sort=False)[col]
        df[f"{col}_lag_1"] = grp_s.shift(1)
        for w in STATUS_ROLLS:
            shifted_s = grp_s.shift(1)
            df[f"{col}_roll_mean_{w}"] = (
                shifted_s.rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_sum_{w}"] = (
                shifted_s.rolling(w, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )

    wh_ts = (
        df.groupby(["office_from_id", "timestamp"], as_index=False)[target_col]
        .agg(wh_target_sum="sum", wh_target_mean="mean")
        .sort_values(["office_from_id", "timestamp"])
    )
    grp_wh_sum = wh_ts.groupby("office_from_id", sort=False)["wh_target_sum"]
    grp_wh_mean = wh_ts.groupby("office_from_id", sort=False)["wh_target_mean"]

    wh_ts["wh_target_sum_lag_1"] = grp_wh_sum.shift(1)
    wh_ts["wh_target_sum_lag_2"] = grp_wh_sum.shift(2)
    wh_ts["wh_target_mean_lag_1"] = grp_wh_mean.shift(1)

    for w in WH_ROLLS:
        shifted_wh = grp_wh_sum.shift(1)
        wh_ts[f"wh_target_roll_mean_{w}"] = (
            shifted_wh.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        wh_ts[f"wh_target_roll_sum_{w}"] = (
            shifted_wh.rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        wh_ts[f"wh_target_roll_std_{w}"] = (
            shifted_wh.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
        )

    df = df.merge(wh_ts, on=["office_from_id", "timestamp"], how="left")

    global_ts = (
        df.groupby("timestamp", as_index=False)[target_col]
        .agg(global_target_sum="sum", global_target_mean="mean")
        .sort_values("timestamp")
    )
    global_ts["global_target_sum_lag_1"] = global_ts["global_target_sum"].shift(1)
    global_ts["global_target_mean_lag_1"] = global_ts["global_target_mean"].shift(1)
    for w in GLOBAL_ROLLS:
        global_ts[f"global_target_roll_mean_{w}"] = (
            global_ts["global_target_sum"].shift(1).rolling(w, min_periods=1).mean()
        )
        global_ts[f"global_target_roll_sum_{w}"] = (
            global_ts["global_target_sum"].shift(1).rolling(w, min_periods=1).sum()
        )
        global_ts[f"global_target_roll_std_{w}"] = (
            global_ts["global_target_sum"].shift(1).rolling(w, min_periods=2).std()
        )
    df = df.merge(global_ts, on="timestamp", how="left")

    df["route_share_lag_1"] = df["target_lag_1"] / (df["wh_target_sum_lag_1"] + 1e-6)
    df["route_share_lag_2"] = df["target_lag_2"] / (df["wh_target_sum_lag_2"] + 1e-6)
    df["route_global_share_lag_1"] = df["target_lag_1"] / (
        df["global_target_sum_lag_1"] + 1e-6
    )
    df["route_minus_wh_mean_lag_1"] = df["target_lag_1"] - df["wh_target_mean_lag_1"]
    df["route_to_wh_mean_ratio_lag_1"] = df["target_lag_1"] / (
        df["wh_target_mean_lag_1"] + 1e-6
    )

    for lag in SHARE_LAGS:
        if f"target_lag_{lag}" in df.columns:
            df[f"route_share_lag_{lag}"] = df[f"target_lag_{lag}"] / (
                df["wh_target_sum_lag_1"] + 1e-6
            )

    return df
