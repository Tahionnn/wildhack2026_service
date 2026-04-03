# models.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# Represents a single row from the incoming CSV file.
class CSVRow(BaseModel):
    office_from_id: str
    period_start: datetime
    call_by: Optional[str] = None
    forecasted_demand: float
    trucks_to_call: int
    capacity_provided: float
    shortage: float
    excess_capacity: float
    fill_rate_pct: float
    status: str
    solver_status: str

# Message models for the three target services.
# Adapt these fields to match the actual requirements of your TMS, WMS, and YMS.
class TMSMessage(BaseModel):
    office_id: str
    period_start: datetime
    trucks_to_call: int
    status: str

class WMSMessage(BaseModel):
    office_id: str
    period_start: datetime
    forecasted_demand: float
    capacity_provided: float
    fill_rate_pct: float

class YMSMessage(BaseModel):
    office_id: str
    period_start: datetime
    shortage: float
    excess_capacity: float