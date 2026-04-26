from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import pandas as pd
import io

# Importing your logic modules
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

# ── Request Models (for JSON endpoints) ────────────────────────────────────────

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

# ── JSON Endpoints (Existing Logic) ───────────────────────────────────────────

@app.post("/audit", tags=["JSON Endpoints"])
async def run_audit(req: AuditRequest):
    try:
        df = pd.DataFrame(req.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty.")
        
        result = AuditEngine().calculate_dir(
            df=df, target=req.target, protected_col=req.protected_col,
            unp_val=req.unprivileged_val, pri_val=req.privileged_val, pos_val=req.positive_val
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/proxy-check", tags=["JSON Endpoints"])
async def proxy_check(req: ProxyRequest):
    try:
        df = pd.DataFrame(req.data)
        result = ProxyDetector().detect_proxies(
            df=df, protected_col=req.protected_col, threshold=req.threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/repair", tags=["JSON Endpoints"])
async def repair_data(req: RepairRequest):
    try:
        df = pd.DataFrame(req.data)
        repaired_df = SyntheticRepairEngine().generate_fair_data(
            df=df, target=req.target, protected_col=req.protected_col
        )
        return repaired_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── CSV/File Upload Endpoints (New Logic) ──────────────────────────────────────

@app.post("/audit-csv", tags=["CSV Upload"])
async def audit_csv(
    file: UploadFile = File(...),
    target: str = Form(...),
    protected_col: str = Form(...),
    unprivileged_val: str = Form(...),
    privileged_val: str = Form(...),
    positive_val: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        result = AuditEngine().calculate_dir(
            df=df, target=target, protected_col=protected_col,
            unp_val=unprivileged_val, pri_val=privileged_val, pos_val=positive_val
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV Audit failed: {str(e)}")

@app.post("/proxy-check-csv", tags=["CSV Upload"])
async def proxy_check_csv(
    file: UploadFile = File(...),
    protected_col: str = Form(...),
    threshold: float = Form(0.05)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        result = ProxyDetector().detect_proxies(
            df=df, protected_col=protected_col, threshold=threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV Proxy check failed: {str(e)}")

@app.post("/repair-csv", tags=["CSV Upload"])
async def repair_csv(
    file: UploadFile = File(...),
    target: str = Form(...),
    protected_col: str = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        repaired_df = SyntheticRepairEngine().generate_fair_data(
            df=df, target=target, protected_col=protected_col
        )
        # We return the repaired data as a list of dicts so it's viewable in browser
        return repaired_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV Repair failed: {str(e)}")
