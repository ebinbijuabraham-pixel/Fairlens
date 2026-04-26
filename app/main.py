from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import pandas as pd

from .audit import AuditEngine
from .proxy_detector import ProxyDetector
from .remediation import SyntheticRepairEngine

app = FastAPI(
    title="FairLens FinTech Compliance API",
    description="Bias detection, proxy analysis, and data remediation for loan approval models.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ─────────────────────────────────────────────────────────────

class AuditRequest(BaseModel):
    data: List[dict]
    target: str
    protected_col: str
    unprivileged_val: Any
    privileged_val: Any
    positive_val: Optional[Any] = None


class ProxyRequest(BaseModel):
    data: List[dict]
    protected_col: str
    threshold: Optional[float] = 0.05


class RepairRequest(BaseModel):
    data: List[dict]
    target: str
    protected_col: str


# ── Health Check ───────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {"status": "FairLens API is running", "version": "2.0.0"}


# ── Audit Endpoint ─────────────────────────────────────────────────────────────

@app.post("/audit")
async def run_audit(req: AuditRequest):
    try:
        df = pd.DataFrame(req.data)

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")
        if req.target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{req.target}' not found. Available: {list(df.columns)}",
            )
        if req.protected_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Protected column '{req.protected_col}' not found. Available: {list(df.columns)}",
            )

        result = AuditEngine().calculate_dir(
            df=df,
            target=req.target,
            protected_col=req.protected_col,
            unp_val=req.unprivileged_val,
            pri_val=req.privileged_val,
            pos_val=req.positive_val,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


# ── Proxy Check Endpoint ───────────────────────────────────────────────────────

@app.post("/proxy-check")
async def proxy_check(req: ProxyRequest):
    try:
        df = pd.DataFrame(req.data)

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")
        if req.protected_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Protected column '{req.protected_col}' not found. Available: {list(df.columns)}",
            )

        result = ProxyDetector().detect_proxies(
            df=df,
            protected_col=req.protected_col,
            threshold=req.threshold,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy check failed: {str(e)}")


# ── Repair Endpoint ────────────────────────────────────────────────────────────

@app.post("/repair")
async def repair_data(req: RepairRequest):
    try:
        df = pd.DataFrame(req.data)

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

        repaired_df = SyntheticRepairEngine().generate_fair_data(
            df=df,
            target=req.target,
            protected_col=req.protected_col,
        )
        return repaired_df.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Repair failed: {str(e)}")
