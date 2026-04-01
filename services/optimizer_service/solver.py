"""
truck_dispatcher.py
===================
Solves the truck dispatching LP:

    min  Σ_{w,t} [ α·x_{w,t} + β·u_{w,t} + γ·e_{w,t} ]

    s.t. C·x_{w,t} + e_{w,t} - u_{w,t} = D_{w,t}   ∀ w,t   (capacity balance)
         x_{w,t} ≤ N_max[w]                           ∀ w,t   (fleet limit)
         x_{w,t} ∈ Z≥0,  u,e ≥ 0

Business assumptions:
  A1  All trucks have the same capacity C (default 33 containers).
  A2  Lead time is fixed: trucks must be called T_lead hours before the period.
  A3  Demand is aggregated at warehouse level (routes share the same truck pool).
  A4  Fleet cap per warehouse is configurable; unconstrained if None.
  A5  Penalty β >> α so shortages are strongly discouraged.

Integration:
  Call `optimize_dispatch(forecast_df, cfg)`.
  Returns a DataFrame with one row per (warehouse, period) and columns:
    trucks_to_call, call_by, shortage, excess_capacity, status
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import pandas as pd
import pulp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DispatchConfig:
    """All tunable parameters in one place — easy to load from env / YAML."""

    # Physical
    truck_capacity: int = 33          # containers per truck (assumption A1)
    lead_time_hours: float = 4.0      # hours before period start to call truck (A2)

    # Cost weights
    cost_per_truck: float = 1.0       # α — normalised unit cost
    penalty_shortage: float = 50.0   # β — per unmet container (SLA violation)
    penalty_excess: float = 0.05     # γ — per idle container-slot (small)

    # Fleet constraints  {office_from_id: max_trucks}
    # None means unconstrained for that warehouse.
    max_trucks_per_warehouse: dict[str | int, int] = field(default_factory=dict)

    # Global fleet cap across ALL warehouses per period (None = unconstrained)
    max_trucks_global: Optional[int] = None

    # Solver
    solver_time_limit_seconds: int = 60
    mip_gap: float = 0.01            # 1 % optimality gap is fine for ops


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def optimize_dispatch(
    forecast_df: pd.DataFrame,
    cfg: DispatchConfig | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    forecast_df : DataFrame with columns
        - route_id          : unique route identifier
        - office_from_id    : warehouse id (string or int)
        - timestamp         : period start (datetime-like)
        - predicted_demand  : predicted target_2h for this route & period

    cfg : DispatchConfig (uses defaults if None)

    Returns
    -------
    DataFrame with columns:
        office_from_id, period_start, call_by,
        forecasted_demand, trucks_to_call, capacity_provided,
        shortage, excess_capacity, fill_rate_pct, status
    """
    if cfg is None:
        cfg = DispatchConfig()

    _validate_input(forecast_df)

    # ------------------------------------------------------------------
    # 1. Aggregate demand per (warehouse, period)
    # ------------------------------------------------------------------
    forecast_df = forecast_df.copy()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

    warehouse_demand: pd.DataFrame = (
        forecast_df
        .groupby(["office_from_id", "timestamp"], sort=True)["predicted_demand"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_demand": "D", "timestamp": "period_start"})
    )

    warehouses = warehouse_demand["office_from_id"].unique().tolist()
    periods = sorted(warehouse_demand["period_start"].unique())

    # Index D_{w,t} for fast look-up
    demand_index: dict[tuple, float] = {
        (row.office_from_id, row.period_start): row.D
        for row in warehouse_demand.itertuples()
    }

    # ------------------------------------------------------------------
    # 2. Build LP
    # ------------------------------------------------------------------
    prob = pulp.LpProblem("truck_dispatch", pulp.LpMinimize)

    # Decision variables
    x: dict[tuple, pulp.LpVariable] = {}   # trucks (integer)
    u: dict[tuple, pulp.LpVariable] = {}   # shortage (continuous)
    e: dict[tuple, pulp.LpVariable] = {}   # excess   (continuous)

    for w in warehouses:
        n_max = cfg.max_trucks_per_warehouse.get(w)
        for t in periods:
            key = (w, t)
            x[key] = pulp.LpVariable(
                f"x_{w}_{t}",
                lowBound=0,
                upBound=n_max,          # None → unconstrained
                cat=pulp.constants.LpInteger,
            )
            u[key] = pulp.LpVariable(f"u_{w}_{t}", lowBound=0, cat=pulp.constants.LpContinuous)
            e[key] = pulp.LpVariable(f"e_{w}_{t}", lowBound=0, cat=pulp.constants.LpContinuous)

    # Objective
    prob += pulp.lpSum(
        cfg.cost_per_truck * x[w, t]
        + cfg.penalty_shortage * u[w, t]
        + cfg.penalty_excess * e[w, t]
        for w in warehouses
        for t in periods
    )

    # Constraints
    for w in warehouses:
        for t in periods:
            D = demand_index.get((w, t), 0.0)
            # (1) Capacity balance
            prob += (
                cfg.truck_capacity * x[w, t] + e[w, t] - u[w, t] == D,
                f"balance_{w}_{t}",
            )

    # (2) Global fleet cap per period
    if cfg.max_trucks_global is not None:
        for t in periods:
            prob += (
                pulp.lpSum(x[w, t] for w in warehouses) <= cfg.max_trucks_global,
                f"global_cap_{t}",
            )

    # ------------------------------------------------------------------
    # 3. Solve
    # ------------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(
        msg=0,
        timeLimit=cfg.solver_time_limit_seconds,
        gapRel=cfg.mip_gap,
    )
    status_code = prob.solve(solver)
    solve_status = pulp.LpStatus[status_code]

    if solve_status not in ("Optimal", "Not Solved"):
        logger.warning("Solver returned status: %s", solve_status)

    logger.info(
        "LP solved | status=%s | objective=%.2f",
        solve_status,
        pulp.value(prob.objective) or 0.0,
    )

    # ------------------------------------------------------------------
    # 4. Extract results
    # ------------------------------------------------------------------
    lead_delta = timedelta(hours=cfg.lead_time_hours)
    rows = []

    for w in warehouses:
        for t in periods:
            D = demand_index.get((w, t), 0.0)
            trucks = int(round(pulp.value(x[w, t]) or 0))
            shortage = max(pulp.value(u[w, t]) or 0.0, 0.0)
            excess   = max(pulp.value(e[w, t]) or 0.0, 0.0)
            capacity = trucks * cfg.truck_capacity

            fill_rate = (capacity - excess) / D * 100 if D > 0 else 100.0

            row_status = _row_status(shortage, excess, D)

            rows.append({
                "office_from_id":    w,
                "period_start":      t,
                "call_by":           t - lead_delta,
                "forecasted_demand": round(D, 2),
                "trucks_to_call":    trucks,
                "capacity_provided": capacity,
                "shortage":          round(shortage, 2),
                "excess_capacity":   round(excess, 2),
                "fill_rate_pct":     round(fill_rate, 1),
                "status":            row_status,
                "solver_status":     solve_status,
            })

    result_df = pd.DataFrame(rows).sort_values(["office_from_id", "period_start"])
    return result_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# FastAPI integration helper
# ---------------------------------------------------------------------------

def dispatch_response_for_warehouse(
    result_df: pd.DataFrame,
    warehouse_id: str | int,
    as_of: pd.Timestamp | None = None,
) -> list[dict]:
    """
    Filter results for a specific warehouse and optionally only future periods.
    Returns a list of dicts suitable for JSON serialisation in a REST response.
    """
    df = result_df[result_df["office_from_id"] == warehouse_id].copy()
    if as_of is not None:
        df = df[df["period_start"] >= as_of]
    df["period_start"] = df["period_start"].dt.isoformat()
    df["call_by"]      = df["call_by"].dt.isoformat()
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_input(df: pd.DataFrame) -> None:
    required = {"route_id", "office_from_id", "timestamp", "predicted_demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"forecast_df is missing columns: {missing}")
    if df.empty:
        raise ValueError("forecast_df must not be empty")
    if df["predicted_demand"].isna().any():
        raise ValueError("predicted_demand contains NaN values")


def _row_status(shortage: float, excess: float, demand: float) -> str:
    if demand == 0:
        return "NO_DEMAND"
    if shortage > 0:
        return "SHORTAGE"
    if excess > demand * 0.5:
        return "OVERALLOCATED"
    return "OK"