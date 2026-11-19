"""
Climate delta-prediction inference service.

Loads the pre-trained model bundles (model_bundles/) once at startup, and exposes
endpoints for the NestJS backend to call. Does NOT retrain anything - inference only.

IMPORTANT - single bias-correction method per variable:
Each variable uses exactly ONE bias-correction method, chosen as whichever method
scored better in our own analysis (see consolidated_results.csv):
    Precipitation -> Linear Scaling (LS)
    Tmax          -> Linear Scaling (LS)
    Tmin          -> Variance Scaling (VS)
    Tmean         -> Linear Scaling (LS)
This means uploaded data MUST be corrected using that specific method - a file
corrected a different way will not error, it will just produce an incorrect
prediction, because the model was trained on that method's numeric distribution.
CanESM5 has no LS temperature data, so it is not available for Tmax or Tmean
(only Precipitation and Tmin).

Granularity - every endpoint that runs a model takes a `granularity`:
daily/monthly/yearly, each backed by its OWN separately trained bundle (a monthly
bundle learns month-over-month change, not day-over-day resampled after the fact).
Uploaded files must already BE at the selected granularity (one row per day, month,
or year, matching whichever you pick) - the service does not resample an upload.
Server-side reference-data endpoints (/predict/ensemble, /envelope) DO resample their
own (always-daily) reference CSVs down to the requested granularity, since that data
is under our control. Yearly bundles are trained on only ~22 historical data points
(1993-2014) - treat yearly predictions as much lower-confidence than daily/monthly,
that's a real data-volume limitation, not a bug (see model_bundles/*_yearly/metadata.json
test_r2_by_model for how noisy this gets).

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health                    - liveness check
    GET  /models                    - list available GCM x variable bundles (the registry)
    POST /predict                   - one GCM, one model, one uploaded CSV
    POST /predict/compare_models    - one GCM, all 3 models, one uploaded CSV
    GET  /predict/ensemble          - all GCMs' own reference data, individual + ensemble
    POST /predict/upload_ensemble   - all GCMs run on ONE uploaded CSV, individual + ensemble
"""
import io
import json
import os
import re
from enum import Enum
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

SERVICE_DESCRIPTION = """
Each variable uses exactly ONE bias-correction method - uploaded data must be
corrected that way for predictions to be meaningful:

| Variable | Required method |
|---|---|
| Precipitation | Linear Scaling (LS) |
| Tmax | Linear Scaling (LS) |
| Tmin | Variance Scaling (VS) |
| Tmean | Linear Scaling (LS) |

CanESM5 has no linear-scaling temperature data, so it is unavailable for Tmax and Tmean
(only Precipitation and Tmin). See GET /models for the exact list of what's available.
"""


class GCM(str, Enum):
    MIROC = "MIROC"
    CanESM5 = "CanESM5"
    MPI = "MPI"
    CNRM = "CNRM"


class Variable(str, Enum):
    Precipitation = "Precipitation"
    Tmax = "Tmax"
    Tmin = "Tmin"
    Tmean = "Tmean"


class ModelChoice(str, Enum):
    rf = "rf"
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    ridge = "ridge"  # opt-in only - never auto-selected by "best". See note below.
    best = "best"  # not a fixed model - picks whichever of rf/xgboost/lightgbm tested best for that specific bundle
    # Ridge is available as an explicit choice, but deliberately excluded from what "best"
    # can resolve to (BEST_MODEL_PER_VARIABLE only ever averages rf/xgboost/lightgbm scores -
    # see load_all_bundles). Reason: unlike the tree ensembles, Ridge extrapolates past the
    # training input range instead of flatlining - useful for the yearly-flatline symptom,
    # but /predict/upload_ensemble runs ONE uploaded file through EVERY GCM's model,
    # including GCMs whose own training range the file's values never resembled. A GCM's
    # Ridge model only ever got extrapolation-safety-checked (at training time, see
    # export_granularity_bundles.py) against ITS OWN native SSP range, so feeding it another
    # GCM's absolute values can still blow past that (MPI's model once hit +64 on CNRM's
    # real uploaded data - worse than the tree flatline it was meant to fix). Since a user
    # can now pick Ridge explicitly, _reliability() re-checks every actual Ridge prediction
    # against the bundle's own ridge_plausibility_cap (computed once at training time and
    # stored in metadata.json) and flags reliable=false with reason="extrapolation_risk" if
    # this specific request's output exceeded it - this catches a live blowup the way the
    # R²-based check catches a low-skill model, instead of a blanket ban on the algorithm.


class Granularity(str, Enum):
    daily = "daily"
    monthly = "monthly"
    yearly = "yearly"


# Reference CSVs (reference_data/, always daily) get resampled to this frequency before
# being fed to a monthly/yearly bundle - None means "use the raw daily rows as-is".
GRANULARITY_FREQ = {"daily": None, "monthly": "ME", "yearly": "YE"}

# Final output resampling for charts, applied uniformly regardless of data source. For
# daily bundles this aggregates noisy day-level predictions up to annual points (matches
# the original single-granularity behavior). For monthly/yearly bundles, the rows are
# already at that resolution, so this is an idempotent no-op - kept for one uniform code
# path instead of branching per granularity at every call site.
DISPLAY_FREQ = {"daily": "YE", "monthly": "ME", "yearly": "YE"}


BUNDLES_DIR = os.environ.get("MODEL_BUNDLES_DIR", os.path.join(os.path.dirname(__file__), "..", "model_bundles"))
REFERENCE_DIR = os.path.join(BUNDLES_DIR, "reference_data")

# Which reference CSV backs each (gcm, variable) - used ONLY by /predict/ensemble, which
# compares all GCMs' own reference projections rather than a user upload (a public user
# only ever has one GCM's data; an ensemble across GCMs needs all of them at once).
REFERENCE_FILES = {
    ("MIROC", "Precipitation"): "MIROC-pcp-daily-linear_scaling-location-1.csv",
    ("CanESM5", "Precipitation"): "CanEsm5-pcp-daily-linear_scaling-location-1.csv",
    ("MPI", "Precipitation"): "MPI-pcp-daily-linear_scaling-location-1.csv",
    ("CNRM", "Precipitation"): "CNRM-pcp-daily-linear_scaling-location-1.csv",
    ("CNRM", "Tmax"): "CNRM-Tmax-daily-linear_scaling-location-1 (1).csv",
    ("MIROC", "Tmax"): "MIROC-Tmax-daily-linear_scaling-location-1 (2).csv",
    ("MPI", "Tmax"): "MPI-Tmax-daily-linear_scaling-location-1.csv",
    ("CNRM", "Tmin"): "CNRM-Tmin-daily-variance-scaling-location-1.csv",
    ("MIROC", "Tmin"): "MIROC-Tmin-daily-variance-scaling-location-1.csv",
    ("MPI", "Tmin"): "MPI-Tmin-daily-variance-scaling-location-1.csv",
    ("CanESM5", "Tmin"): "CanESM-Tmin-daily-variance-scaling-location-1.csv",
    ("CNRM", "Tmean"): "CNRM-Tmean-daily-linear_scaling-derived-location-1.csv",
    ("MIROC", "Tmean"): "MIROC-Tmean-daily-linear_scaling-derived-location-1.csv",
    ("MPI", "Tmean"): "MPI-Tmean-daily-linear_scaling-derived-location-1.csv",
}

app = FastAPI(title="Climate Delta Inference Service", description=SERVICE_DESCRIPTION)

# Comma-separated list of allowed origins, e.g. "https://app.example.com,https://admin.example.com".
# Defaults to "*" (any origin) if unset - set ALLOWED_ORIGINS in production.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _allowed_origins == "*" else [o.strip() for o in _allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Registry: loaded once at startup, kept warm in memory ----
REGISTRY = {}  # key: "GCM_Variable" -> {models, scaler, schema, metadata}


def _load_bundle(bundle_dir: str) -> dict:
    with open(os.path.join(bundle_dir, "schema.json")) as f:
        schema = json.load(f)
    with open(os.path.join(bundle_dir, "metadata.json")) as f:
        metadata = json.load(f)
    models = {
        "rf": joblib.load(os.path.join(bundle_dir, "model_rf.joblib")),
        "xgboost": joblib.load(os.path.join(bundle_dir, "model_xgboost.joblib")),
        "lightgbm": joblib.load(os.path.join(bundle_dir, "model_lightgbm.joblib")),
        # Loaded but never auto-selected by "best" (see ModelChoice's comment) - only reachable
        # by explicitly passing model=ridge, at which point _reliability() checks its output
        # against this bundle's own ridge_plausibility_cap before calling it trustworthy.
        "ridge": joblib.load(os.path.join(bundle_dir, "model_ridge.joblib")),
    }
    scaler = joblib.load(os.path.join(bundle_dir, "scaler.joblib"))
    return {"models": models, "scaler": scaler, "schema": schema, "metadata": metadata}


BEST_MODEL_PER_VARIABLE = {}  # {variable: {granularity: "rf"/"xgboost"/"lightgbm"}}, picked once for all GCMs


@app.on_event("startup")
def load_all_bundles():
    manifest_path = os.path.join(BUNDLES_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"manifest.json not found at {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    for entry in manifest:
        key = f"{entry['gcm']}_{entry['variable']}_{entry['granularity']}"
        bundle_dir = os.path.join(BUNDLES_DIR, key)
        REGISTRY[key] = {**_load_bundle(bundle_dir), "manifest_entry": entry}
    print(f"Loaded {len(REGISTRY)} model bundles: {list(REGISTRY.keys())}")

    # "Best model" is picked ONCE per (variable, granularity) - not per GCM: average each
    # algorithm's held-out test R2 (metadata["test_r2_by_model"]) across every GCM that has
    # that variable+granularity, then take the algorithm with the highest average. The
    # winner can differ by granularity, not just by variable (e.g. daily Tmax may favor a
    # different algorithm than yearly Tmax), since each granularity is a separately trained
    # bundle with its own data volume and noise characteristics.
    by_key = {}
    for bundle in REGISTRY.values():
        variable = bundle["metadata"]["variable"]
        granularity = bundle["metadata"]["granularity"]
        scores = bundle["metadata"].get("test_r2_by_model")
        if not scores:
            continue
        by_key.setdefault((variable, granularity), []).append(scores)
    for (variable, granularity), score_list in by_key.items():
        avg = {m: sum(s[m] for s in score_list) / len(score_list) for m in ("rf", "xgboost", "lightgbm")}
        BEST_MODEL_PER_VARIABLE.setdefault(variable, {})[granularity] = max(avg, key=avg.get)
    print(f"Best model per variable/granularity: {BEST_MODEL_PER_VARIABLE}")


class ModelInfo(BaseModel):
    gcm: str
    variable: str
    granularity: str
    method: str
    models: List[str] = ["rf", "xgboost", "lightgbm", "ridge"]


@app.get("/health")
def health():
    return {"status": "ok", "bundles_loaded": len(REGISTRY)}


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    return [entry["manifest_entry"] for entry in REGISTRY.values()]


@app.get("/best_model_per_variable")
def best_model_per_variable():
    """Which single algorithm (rf/xgboost/lightgbm) model="best" resolves to, per variable -
    lets a frontend show the real name (e.g. "Best Model (XGBoost)") before any prediction
    is run, instead of a generic placeholder."""
    return BEST_MODEL_PER_VARIABLE


# ---- Preprocessing (mirrors the training pipeline exactly) ----
REQUIRED_RAW_COLUMNS = [
    "Year", "Historical-Ref", "Historical-10th", "Historical-90th",
    "SSP-4.5-Ref", "SSP-4.5-10th", "SSP-4.5-90th",
    "SSP-8.5-Ref", "SSP-8.5-10th", "SSP-8.5-90th",
]


_GAP_FILL_FREQ = {"daily": "D", "monthly": "MS", "yearly": "YS"}


def _detect_granularity(dates: pd.Series) -> str:
    """Infer daily/monthly/yearly from the median gap between consecutive (non-null) row
    dates - lets an upload pick its own bundle automatically instead of the user having to
    declare it. Reference-data endpoints don't use this (they pass an explicit granularity
    to choose an output resolution from always-daily source data - a different question)."""
    d = dates.dropna().sort_values().unique()
    if len(d) < 2:
        return "yearly"
    median_gap = pd.Series(d).diff().dropna().dt.days.median()
    if median_gap <= 3:
        return "daily"
    if median_gap <= 45:
        return "monthly"
    return "yearly"


def _read_table(file_bytes: bytes, filename: str = None) -> pd.DataFrame:
    """Parses either CSV or Excel (.xlsx/.xls) bytes into a DataFrame. Dispatches on the
    upload's filename extension when available; if it's missing or unrecognized, falls back
    to sniffing the leading bytes (.xlsx/.xls are zip/OLE containers with a distinctive magic
    number, CSV is plain text) so a mislabeled or extension-less upload still works."""
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    if ext not in (".csv", ".xlsx", ".xls"):
        if file_bytes[:4] == b"PK\x03\x04" or file_bytes[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            ext = ".xlsx"
        else:
            ext = ".csv"

    if ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read uploaded Excel file: {e}")
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded CSV file: {e}")


_ALIAS_SCOPE_85 = re.compile(r"8\.?5")
_ALIAS_SCOPE_45 = re.compile(r"4\.?5")
_ALIAS_90TH = re.compile(r"90\s*th|90\s*%|90\s*percentile", re.IGNORECASE)
_ALIAS_10TH = re.compile(r"10\s*th|10\s*%|10\s*percentile", re.IGNORECASE)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort remap of loosely-named columns onto the canonical schema this service
    expects. Handles human-oriented exports meant for plotting rather than machine
    ingestion - e.g. 'Hist. Ref. Pre., 1950-2014', 'SSP2-4.5 ', 'ssp4.590th' - by guessing
    each column's scenario (SSP-4.5 / SSP-8.5 / Historical, from a '4.5'/'8.5' substring,
    defaulting to Historical when neither is present) and value type (Ref / 10th / 90th,
    from a '10th'/'90th' substring). Only touches columns that aren't already a canonical
    name, and never claims a canonical target that's already taken (by an existing column
    or an earlier match in this same pass) or that isn't one of REQUIRED_RAW_COLUMNS - so a
    well-formed file, and the Year/Date column, are never altered. Some exports use
    'Category' as the header for what is otherwise a plain year column (e.g. the annual
    Fullplot spreadsheets) - that gets remapped to 'Year' rather than falling through to
    the Ref/10th/90th guesser below, which would otherwise mistake it for a data column."""
    rename = {}
    used_targets = set(df.columns) & set(REQUIRED_RAW_COLUMNS)
    for col in df.columns:
        if col in REQUIRED_RAW_COLUMNS:
            continue
        label = str(col).strip()
        if label.lower() in ("year", "date"):
            continue
        if label.lower() == "category" and "Year" not in df.columns and "Year" not in used_targets:
            rename[col] = "Year"
            used_targets.add("Year")
            continue
        if _ALIAS_90TH.search(label):
            kind = "90th"
        elif _ALIAS_10TH.search(label):
            kind = "10th"
        else:
            kind = "Ref"
        if _ALIAS_SCOPE_85.search(label):
            scope = "SSP-8.5"
        elif _ALIAS_SCOPE_45.search(label):
            scope = "SSP-4.5"
        else:
            scope = "Historical"
        canonical = f"{scope}-{kind}"
        if canonical in REQUIRED_RAW_COLUMNS and canonical not in used_targets:
            rename[col] = canonical
            used_targets.add(canonical)
    return df.rename(columns=rename) if rename else df


def _read_and_clean(csv_bytes: bytes, is_precip: bool, granularity: str = None, filename: str = None) -> pd.DataFrame:
    """`granularity=None` (the upload-endpoint case) auto-detects from the file's own date
    spacing and stashes it on `df.attrs["granularity"]`. Reference-data endpoints pass an
    explicit granularity to override this, since they choose an output resolution rather
    than detect one. Accepts CSV or Excel (.xlsx/.xls) input - see `_read_table`. Column
    names are normalized onto the canonical schema before validation - see
    `_normalize_columns` - so loosely-named human exports don't have to be hand-edited."""
    df = _read_table(csv_bytes, filename)
    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Uploaded CSV is missing required columns: {missing}")

    if pd.api.types.is_numeric_dtype(df["Year"]):
        # Bare year values (e.g. a genuinely yearly file: 1950, 1951, ...) - pd.to_datetime
        # on raw integers/floats interprets them as nanoseconds since epoch, not calendar
        # years, silently collapsing every row to ~1970. Parse as an explicit %Y string
        # instead so bare-year "Year" columns land on the correct Jan-1 date.
        df["Date"] = pd.to_datetime(df["Year"].astype("Int64").astype(str), format="%Y", errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df["Year"], errors="coerce")

    if granularity is None:
        granularity = _detect_granularity(df["Date"])
    df.attrs["granularity"] = granularity

    if df["Date"].isna().any():
        last_valid = df["Date"].dropna().max()
        n_missing = df["Date"].isna().sum()
        if pd.isna(last_valid):
            raise HTTPException(status_code=400, detail="Could not parse any dates from the 'Year' column.")
        gen = pd.date_range(start=last_valid + pd.Timedelta(days=1), periods=n_missing, freq=_GAP_FILL_FREQ[granularity])
        df.loc[df["Date"].isna(), "Date"] = gen
    df["YEAR"] = df["Date"].dt.year

    if is_precip:
        num_cols = [c for c in REQUIRED_RAW_COLUMNS[1:] if df[c].dtype == "float64"]
        df[num_cols] = df[num_cols].clip(lower=0)
        for col in [c for c in REQUIRED_RAW_COLUMNS if c.startswith("SSP")]:
            df.loc[df[col] > 80, col] = 80

    return df


def _split_hist_ssp(df: pd.DataFrame):
    # Sorted explicitly rather than assumed - every _delta column downstream is a
    # `.diff()` over these rows, which silently computes garbage (a difference between
    # two non-adjacent dates) if the input wasn't already chronological. Every file used
    # so far happens to be pre-sorted, so this has never been observed to matter in
    # practice, but nothing enforced it either.
    df = df.sort_values("Date").reset_index(drop=True)
    hist = df[df["YEAR"] <= 2014][["Date", "YEAR", "Historical-Ref", "Historical-10th", "Historical-90th"]].copy()
    ssp = df[df["YEAR"] >= 2015][["Date", "YEAR", "SSP-4.5-Ref", "SSP-4.5-10th", "SSP-4.5-90th",
                                    "SSP-8.5-Ref", "SSP-8.5-10th", "SSP-8.5-90th"]].copy()
    return hist, ssp


def _resample_to_granularity(hist: pd.DataFrame, ssp: pd.DataFrame, granularity: str):
    """reference_data/ is always daily - collapse both hist and ssp to the requested
    granularity's mean before delta/absolute computation, the same resampling
    export_granularity_bundles.py applied before training that granularity's bundle.
    Only used for server-side reference-data endpoints; uploaded files are taken as
    already being at the selected granularity (see module docstring)."""
    freq = GRANULARITY_FREQ[granularity]
    if freq is None:
        return hist, ssp

    def resample_df(df, cols):
        g = df.set_index("Date")[cols].resample(freq).mean().reset_index()
        g["YEAR"] = g["Date"].dt.year
        g[cols] = g[cols].ffill().bfill()
        return g

    hist_cols = ["Historical-Ref", "Historical-10th", "Historical-90th"]
    ssp_cols = ["SSP-4.5-Ref", "SSP-4.5-10th", "SSP-4.5-90th", "SSP-8.5-Ref", "SSP-8.5-10th", "SSP-8.5-90th"]
    return resample_df(hist, hist_cols), resample_df(ssp, ssp_cols)


def _prep_single_method(hist: pd.DataFrame, ssp: pd.DataFrame):
    """Clean + delta-construct a single-method hist/ssp pair. Returns (hist_out, ssp, X_cols, target_cols, prep_X)."""
    ref_cols, c10_cols, c90_cols = ["Historical-Ref"], ["Historical-10th"], ["Historical-90th"]
    fill_cols = ref_cols + c10_cols + c90_cols
    hist[fill_cols] = hist[fill_cols].ffill().bfill()
    hist.loc[hist["Historical-Ref"] < hist["Historical-10th"], "Historical-Ref"] = hist["Historical-10th"]
    hist.loc[hist["Historical-Ref"] > hist["Historical-90th"], "Historical-Ref"] = hist["Historical-90th"]

    for col in ref_cols + c10_cols + c90_cols:
        hist[f"{col}_delta"] = hist[col].diff().fillna(0)

    hist_delta_cols = {f"{c}_delta": hist[f"{c}_delta"] for c in ref_cols + c10_cols + c90_cols}
    hist_out = pd.DataFrame({"date": hist["Date"].dt.strftime("%Y-%m-%d"), **hist_delta_cols})

    X_cols = ref_cols + c10_cols + c90_cols + ["YEAR"]
    target_cols = [f"{c}_delta" for c in ref_cols + c10_cols + c90_cols]

    def prep_X(prefix, scaler):
        cols = [f"{prefix}-{c.split('-', 1)[1]}" if c != "YEAR" else "YEAR" for c in X_cols]
        out = ssp[cols].copy()
        out.columns = X_cols
        return scaler.transform(out)

    return hist_out, ssp, X_cols, target_cols, prep_X


def _resolve_model_name(bundle: dict, model_value: str) -> str:
    """model_value == "best" resolves to the one algorithm (rf/xgboost/lightgbm) with the
    highest average held-out test R² across all GCMs for this bundle's variable+granularity
    (BEST_MODEL_PER_VARIABLE, computed at startup from metadata["test_r2_by_model"]). One
    fixed algorithm per variable and granularity, applied to every GCM. Any other value
    passes through unchanged."""
    if model_value == "best":
        variable = bundle["metadata"]["variable"]
        granularity = bundle["metadata"]["granularity"]
        return BEST_MODEL_PER_VARIABLE.get(variable, {}).get(granularity, "lightgbm")
    return model_value


def _predict(bundle: dict, model_value: str, X):
    """Run inference with the resolved model (see _resolve_model_name)."""
    return bundle["models"][_resolve_model_name(bundle, model_value)].predict(X)


def _reliability(bundle: dict, model_value: str, preds: "np.ndarray | None" = None) -> dict:
    """Surfaces whether this call's prediction should be trusted, so callers (and the UI)
    can warn instead of silently trusting it. Two independent checks feed into `reliable`:

    1. R² check (all models): the resolved model's held-out test R² on this bundle's
       Ref-delta target must be positive - some yearly-granularity bundles score negative
       R² on every candidate model (see export_granularity_bundles.py's docstring on
       data-volume limits at that granularity).
    2. Extrapolation check (ridge only): Ridge doesn't flatline outside its training range
       like the tree models do, it keeps extrapolating linearly - which is exactly why it's
       opt-in (see ModelChoice's comment). `ridge_plausibility_cap` was computed once at
       training time as the largest |delta| Ridge produced across this bundle's OWN full
       SSP-4.5+SSP-8.5 range. If `preds` (this specific call's actual output) exceeds that
       cap, the model is extrapolating further than it was ever safety-checked for - e.g.
       because the uploaded data belongs to a different GCM than this bundle's own. Pass
       `preds` whenever the resolved model is ridge; omit it (or pass None) elsewhere.
    """
    resolved = _resolve_model_name(bundle, model_value)
    test_r2 = bundle["metadata"].get("test_r2_by_model", {}).get(resolved)
    reliable = test_r2 is not None and test_r2 > 0
    reason = None if reliable else "low_r2"
    detail = None

    if resolved == "ridge" and preds is not None:
        cap = bundle["metadata"].get("ridge_plausibility_cap")
        if cap is not None:
            max_abs = float(np.max(np.abs(preds))) if preds.size else 0.0
            if max_abs > cap:
                reliable = False
                reason = "extrapolation_risk"
                detail = (f"Ridge predicted a delta of {max_abs:.2f}, past this bundle's own "
                           f"plausibility cap of {cap:.2f} - it's extrapolating further than it "
                           f"was ever safety-checked for (often means the uploaded data doesn't "
                           f"match this GCM's own value range).")

    return {"model_used": resolved, "test_r2": test_r2, "reliable": reliable, "reason": reason, "detail": detail}


@app.post("/predict")
async def predict(
    gcm: GCM = Form(...),
    variable: Variable = Form(...),
    model: ModelChoice = Form(ModelChoice.lightgbm, description="Which trained model to run inference with"),
    file: UploadFile = File(..., description="CSV or Excel (.xlsx/.xls) file - must be corrected with the method GET /models says this variable needs. Granularity (daily/monthly/yearly) is auto-detected from the file's own row spacing, not selected."),
):
    is_precip = variable.lower().startswith("precip")
    content = await file.read()
    df = _read_and_clean(content, is_precip, filename=file.filename)
    granularity = df.attrs["granularity"]

    key = f"{gcm.value}_{variable.value}_{granularity}"
    if key not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"No model available for gcm={gcm.value}, variable={variable.value}, granularity={granularity} (detected from the file's row spacing). See GET /models.")
    bundle = REGISTRY[key]

    hist, ssp = _split_hist_ssp(df)
    hist_out, ssp, X_cols, target_cols, prep_X = _prep_single_method(hist, ssp)

    pred_45 = _predict(bundle, model.value, prep_X("SSP-4.5", bundle["scaler"]))
    pred_85 = _predict(bundle, model.value, prep_X("SSP-8.5", bundle["scaler"]))
    ssp_out = pd.DataFrame({"date": ssp["Date"].dt.strftime("%Y-%m-%d")})
    for i, col in enumerate(target_cols):
        ssp_out[f"ssp45_{col}"] = pred_45[:, i]
        ssp_out[f"ssp85_{col}"] = pred_85[:, i]

    ref_idx = target_cols.index("Historical-Ref_delta")
    ref_preds = np.concatenate([pred_45[:, ref_idx], pred_85[:, ref_idx]])

    return {
        "gcm": gcm.value,
        "variable": variable.value,
        "granularity": granularity,
        "model": model.value,
        "method": bundle["schema"]["method"],
        "historical": hist_out.to_dict(orient="records"),
        "projected": ssp_out.to_dict(orient="records"),
        "reliability": _reliability(bundle, model.value, ref_preds),
    }


@app.post("/predict/compare_models")
async def predict_compare_models(
    gcm: GCM = Form(...),
    variable: Variable = Form(...),
    file: UploadFile = File(..., description="CSV or Excel (.xlsx/.xls) file - must be corrected with the method GET /models says this variable needs. Granularity (daily/monthly/yearly) is auto-detected from the file's own row spacing, not selected."),
):
    """Same as /predict, but runs all three models (rf, xgboost, lightgbm) in one call."""
    is_precip = variable.lower().startswith("precip")
    content = await file.read()
    df = _read_and_clean(content, is_precip, filename=file.filename)
    granularity = df.attrs["granularity"]

    key = f"{gcm.value}_{variable.value}_{granularity}"
    if key not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"No model available for gcm={gcm.value}, variable={variable.value}, granularity={granularity} (detected from the file's row spacing). See GET /models.")
    bundle = REGISTRY[key]
    hist, ssp = _split_hist_ssp(df)
    hist_out, ssp, X_cols, target_cols, prep_X = _prep_single_method(hist, ssp)

    X45 = prep_X("SSP-4.5", bundle["scaler"])
    X85 = prep_X("SSP-8.5", bundle["scaler"])
    ref_idx = target_cols.index("Historical-Ref_delta")
    ssp_out = pd.DataFrame({"date": ssp["Date"].dt.strftime("%Y-%m-%d")})
    reliability = {}
    for model_name, active_model in bundle["models"].items():
        pred_45 = active_model.predict(X45)
        pred_85 = active_model.predict(X85)
        for i, col in enumerate(target_cols):
            ssp_out[f"{model_name}_ssp45_{col}"] = pred_45[:, i]
            ssp_out[f"{model_name}_ssp85_{col}"] = pred_85[:, i]
        ref_preds = np.concatenate([pred_45[:, ref_idx], pred_85[:, ref_idx]])
        reliability[model_name] = _reliability(bundle, model_name, ref_preds)

    return {
        "gcm": gcm.value,
        "variable": variable.value,
        "granularity": granularity,
        "models": list(bundle["models"].keys()),
        "method": bundle["schema"]["method"],
        "historical": hist_out.to_dict(orient="records"),
        "projected": ssp_out.to_dict(orient="records"),
        "reliability": reliability,
    }


def _run_reference_gcm(gcm: str, variable: str, model_name: str, granularity: str):
    """Run one GCM's own reference data (always daily on disk) through its bundle at the
    requested granularity, resampling first. Returns a delta series at that granularity."""
    key = f"{gcm}_{variable}_{granularity}"
    if key not in REGISTRY:
        return None
    bundle = REGISTRY[key]
    is_precip = variable.lower().startswith("precip")

    fname = REFERENCE_FILES[(gcm, variable)]
    with open(os.path.join(REFERENCE_DIR, fname), "rb") as f:
        content = f.read()
    df = _read_and_clean(content, is_precip)
    hist, ssp = _split_hist_ssp(df)
    hist, ssp = _resample_to_granularity(hist, ssp, granularity)
    hist_out, ssp, X_cols, target_cols, prep_X = _prep_single_method(hist, ssp)

    pred_45 = _predict(bundle, model_name, prep_X("SSP-4.5", bundle["scaler"]))
    pred_85 = _predict(bundle, model_name, prep_X("SSP-8.5", bundle["scaler"]))
    idx = target_cols.index("Historical-Ref_delta")

    freq = DISPLAY_FREQ[granularity]
    hist_dates = pd.to_datetime(hist_out["date"])
    ref_preds = np.concatenate([pred_45[:, idx], pred_85[:, idx]])
    return {
        "historical": pd.Series(hist_out["Historical-Ref_delta"].values, index=hist_dates).resample(freq).mean(),
        "projected_ssp45": pd.Series(pred_45[:, idx], index=ssp["Date"]).resample(freq).mean(),
        "projected_ssp85": pd.Series(pred_85[:, idx], index=ssp["Date"]).resample(freq).mean(),
        "reliability": _reliability(bundle, model_name, ref_preds),
    }


def _build_ensemble_response(per_gcm: dict, variable: str, model_name: str, extra_fields: dict, extrapolating: set = frozenset()):
    """`extrapolating` - GCMs whose reliability check came back reason="extrapolation_risk"
    (see _reliability). Their own column is still returned in full for transparency/
    debugging, but they're excluded from ensemble_mean/std/min/max - a Ridge blowup (seen in
    testing: one GCM predicting a delta of +374 against a same-scale ~2 for the others) would
    otherwise silently drag the whole ensemble average and band up to match it, defeating the
    point of flagging it as unreliable in the first place."""
    gcms = list(per_gcm.keys())
    clean_gcms = [g for g in gcms if g not in extrapolating]

    def build_frame(segment):
        frame = pd.DataFrame({g: per_gcm[g][segment] for g in gcms})
        stat_cols = clean_gcms if clean_gcms else gcms  # never produce an all-NaN ensemble
        frame["ensemble_mean"] = frame[stat_cols].mean(axis=1)
        frame["ensemble_std"] = frame[stat_cols].std(axis=1)
        frame["ensemble_min"] = frame[stat_cols].min(axis=1)
        frame["ensemble_max"] = frame[stat_cols].max(axis=1)
        return frame

    hist_frame = build_frame("historical")
    s45_frame = build_frame("projected_ssp45")
    s85_frame = build_frame("projected_ssp85")

    # Anchor each projected frame's ensemble stats (mean/std/min/max) to the historical
    # endpoint, so the summary lines connect visually - same fix as /envelope. Individual
    # GCM columns are left as NaN at the anchor row since a single connecting point isn't
    # meaningful per-GCM the way it is for the aggregate stats.
    anchor_date = hist_frame.index[-1]
    anchor_stats = hist_frame.loc[anchor_date, ["ensemble_mean", "ensemble_std", "ensemble_min", "ensemble_max"]]

    def anchor(frame):
        anchor_row = pd.DataFrame([anchor_stats], index=[anchor_date])
        return pd.concat([anchor_row, frame])

    s45_frame = anchor(s45_frame)
    s85_frame = anchor(s85_frame)

    def to_records(frame):
        records = []
        for date, row in frame.iterrows():
            rec = {"date": date.strftime("%Y-%m-%d")}
            for g in gcms:
                v = row[g] if g in row.index else None
                rec[g] = None if (v is None or pd.isna(v)) else float(v)
            for stat in ["ensemble_mean", "ensemble_std", "ensemble_min", "ensemble_max"]:
                v = row[stat]
                rec[stat] = None if pd.isna(v) else float(v)
            records.append(rec)
        return records

    return {
        "variable": variable,
        "model": model_name,
        "gcms_included": gcms,
        "historical": to_records(hist_frame),
        "projected_ssp45": to_records(s45_frame),
        "projected_ssp85": to_records(s85_frame),
        **extra_fields,
    }


@app.get("/predict/ensemble")
def predict_ensemble(variable: Variable, model: ModelChoice = ModelChoice.lightgbm, granularity: Granularity = Granularity.daily):
    """All GCMs that have this variable, individually + as a cross-GCM ensemble mean,
    historical + projected, at the requested granularity. Uses each GCM's own reference
    data (always daily on disk), resampled to `granularity` before modeling."""
    per_gcm = {}
    reliability = {}
    for g in GCM:
        result = _run_reference_gcm(g.value, variable.value, model.value, granularity.value)
        if result is not None:
            reliability[g.value] = result.pop("reliability")
            per_gcm[g.value] = result
    if not per_gcm:
        raise HTTPException(status_code=404, detail=f"No GCM has a model for variable={variable.value}, granularity={granularity.value}.")
    extrapolating = {g for g, r in reliability.items() if r.get("reason") == "extrapolation_risk"}
    return _build_ensemble_response(per_gcm, variable.value, model.value, {"granularity": granularity.value, "reliability": reliability}, extrapolating)


def _absolute_segments(hist: pd.DataFrame, ssp: pd.DataFrame, granularity: str):
    """Raw absolute Ref value (no model) + its own 10th/90th band, resampled to
    `granularity`, SSP anchored to the historical endpoint - same shape/anchoring as the
    delta series, keyed by date so it can be merged straight into the same records the
    delta chart uses, instead of being a separate response/fetch."""
    freq = DISPLAY_FREQ[granularity]

    def smooth(s, w=3):
        return s.rolling(w, center=True, min_periods=1).mean()

    h = hist.set_index("Date")
    s = ssp.set_index("Date")
    h_mean = smooth(h["Historical-Ref"].resample(freq).mean())
    h_min = smooth(h["Historical-10th"].resample(freq).mean())
    h_max = smooth(h["Historical-90th"].resample(freq).mean())
    s45_mean = smooth(s["SSP-4.5-Ref"].resample(freq).mean())
    s45_min = smooth(s["SSP-4.5-10th"].resample(freq).mean())
    s45_max = smooth(s["SSP-4.5-90th"].resample(freq).mean())
    s85_mean = smooth(s["SSP-8.5-Ref"].resample(freq).mean())
    s85_min = smooth(s["SSP-8.5-10th"].resample(freq).mean())
    s85_max = smooth(s["SSP-8.5-90th"].resample(freq).mean())

    anchor_date = h_mean.index[-1]

    def anchor(mean_s, lo_s, hi_s):
        m = pd.concat([pd.Series([h_mean.iloc[-1]], index=[anchor_date]), mean_s])
        lo = pd.concat([pd.Series([h_min.iloc[-1]], index=[anchor_date]), lo_s])
        hi = pd.concat([pd.Series([h_max.iloc[-1]], index=[anchor_date]), hi_s])
        return m, lo, hi

    s45_mean, s45_min, s45_max = anchor(s45_mean, s45_min, s45_max)
    s85_mean, s85_min, s85_max = anchor(s85_mean, s85_min, s85_max)

    def to_map(mean_s, lo_s, hi_s):
        return {
            d.strftime("%Y-%m-%d"): (round(float(mean_s[d]), 4), round(float(lo_s[d]), 4), round(float(hi_s[d]), 4))
            for d in mean_s.index
        }

    return to_map(h_mean, h_min, h_max), to_map(s45_mean, s45_min, s45_max), to_map(s85_mean, s85_min, s85_max)


@app.post("/predict/upload_ensemble")
async def predict_upload_ensemble(
    variable: Variable = Form(...),
    model: ModelChoice = Form(ModelChoice.lightgbm),
    file: UploadFile = File(..., description="CSV or Excel (.xlsx/.xls) file - must be corrected with the method GET /models says this variable needs. Granularity (daily/monthly/yearly) is auto-detected from the file's own row spacing, not selected."),
):
    """No GCM selection - runs the SAME uploaded file through every GCM's bundle for this
    variable (granularity auto-detected from the file's own row spacing), and returns each
    GCM's own predicted delta plus the cross-GCM ensemble. Each record is also annotated
    with abs_mean/abs_min/abs_max - the file's own raw absolute value and 10th-90th band
    for that same date, no model involved - so one response can drive a chart showing both
    predicted change and actual values together."""
    is_precip = variable.lower().startswith("precip")
    content = await file.read()
    df = _read_and_clean(content, is_precip, filename=file.filename)
    granularity = df.attrs["granularity"]
    hist, ssp = _split_hist_ssp(df)
    abs_hist, abs_s45, abs_s85 = _absolute_segments(hist, ssp, granularity)

    per_gcm = {}
    reliability = {}
    for g in GCM:
        key = f"{g.value}_{variable.value}_{granularity}"
        if key not in REGISTRY:
            continue
        bundle = REGISTRY[key]
        hist_out, ssp_i, X_cols, target_cols, prep_X = _prep_single_method(hist.copy(), ssp.copy())
        pred_45 = _predict(bundle, model.value, prep_X("SSP-4.5", bundle["scaler"]))
        pred_85 = _predict(bundle, model.value, prep_X("SSP-8.5", bundle["scaler"]))
        idx = target_cols.index("Historical-Ref_delta")
        freq = DISPLAY_FREQ[granularity]
        hist_dates = pd.to_datetime(hist_out["date"])
        per_gcm[g.value] = {
            "historical": pd.Series(hist_out["Historical-Ref_delta"].values, index=hist_dates).resample(freq).mean(),
            "projected_ssp45": pd.Series(pred_45[:, idx], index=ssp_i["Date"]).resample(freq).mean(),
            "projected_ssp85": pd.Series(pred_85[:, idx], index=ssp_i["Date"]).resample(freq).mean(),
        }
        ref_preds = np.concatenate([pred_45[:, idx], pred_85[:, idx]])
        reliability[g.value] = _reliability(bundle, model.value, ref_preds)

    if not per_gcm:
        raise HTTPException(status_code=404, detail=f"No GCM has a model for variable={variable.value}, granularity={granularity} (detected from the file's row spacing).")
    best_model_used = BEST_MODEL_PER_VARIABLE.get(variable.value, {}).get(granularity) if model.value == "best" else None
    extrapolating = {g for g, r in reliability.items() if r.get("reason") == "extrapolation_risk"}
    response = _build_ensemble_response(per_gcm, variable.value, model.value, {"granularity": granularity, "best_model_used": best_model_used}, extrapolating)
    response["reliability"] = reliability

    for records, abs_map in [
        (response["historical"], abs_hist),
        (response["projected_ssp45"], abs_s45),
        (response["projected_ssp85"], abs_s85),
    ]:
        for rec in records:
            hit = abs_map.get(rec["date"])
            if hit is not None:
                rec["abs_mean"], rec["abs_min"], rec["abs_max"] = hit

    return response


# ---------------------------------------------------------------------------
# /envelope: raw bias-corrected ABSOLUTE values (no ML model involved at all) -
# every GCM that has this variable, individually + cross-GCM mean/envelope
# (min-max range), historical seamlessly anchored into the SSP lines so there's
# no visual jump at the 2015 boundary. This is the "clean chart" version -
# see /predict/ensemble and /predict/upload_ensemble for the model-prediction
# (delta) version instead, which is noisier because it reflects real model skill.
# ---------------------------------------------------------------------------

def _load_envelope_gcm(gcm: str, variable: str, granularity: str):
    key = (gcm, variable)
    if key not in REFERENCE_FILES:
        return None
    is_precip = variable.lower().startswith("precip")
    with open(os.path.join(REFERENCE_DIR, REFERENCE_FILES[key]), "rb") as f:
        content = f.read()
    df = _read_and_clean(content, is_precip)
    hist = df[df["YEAR"] <= 2014][["Date", "YEAR", "Historical-Ref", "Historical-10th", "Historical-90th"]].copy()
    ssp = df[df["YEAR"] >= 2015][["Date", "YEAR", "SSP-4.5-Ref", "SSP-4.5-10th", "SSP-4.5-90th",
                                    "SSP-8.5-Ref", "SSP-8.5-10th", "SSP-8.5-90th"]].copy()
    hist, ssp = _resample_to_granularity(hist, ssp, granularity)
    freq = DISPLAY_FREQ[granularity]
    hist = hist.set_index("Date")
    ssp = ssp.set_index("Date")
    return {
        "h_ref": hist["Historical-Ref"].resample(freq).mean(),
        "s45_ref": ssp["SSP-4.5-Ref"].resample(freq).mean(),
        "s85_ref": ssp["SSP-8.5-Ref"].resample(freq).mean(),
    }


@app.get("/envelope")
def envelope(variable: Variable, granularity: Granularity = Granularity.daily):
    """Raw bias-corrected absolute values (NOT a model prediction) - every GCM that
    has this variable, individually + cross-GCM mean and min-max envelope, historical
    seamlessly connected into the projection so there's no jump at 2015, resampled to
    the requested granularity."""
    per_gcm = {}
    for g in GCM:
        result = _load_envelope_gcm(g.value, variable.value, granularity.value)
        if result is not None:
            per_gcm[g.value] = result
    if not per_gcm:
        raise HTTPException(status_code=404, detail=f"No reference data for variable={variable.value}.")

    gcms = list(per_gcm.keys())

    def smooth(s, w=3):
        return s.rolling(w, center=True, min_periods=1).mean()

    h_df = pd.DataFrame({g: per_gcm[g]["h_ref"] for g in gcms})
    s45_df = pd.DataFrame({g: per_gcm[g]["s45_ref"] for g in gcms})
    s85_df = pd.DataFrame({g: per_gcm[g]["s85_ref"] for g in gcms})

    h_mean, h_min, h_max = smooth(h_df.mean(axis=1)), smooth(h_df.min(axis=1)), smooth(h_df.max(axis=1))
    s45_mean, s45_min, s45_max = smooth(s45_df.mean(axis=1)), smooth(s45_df.min(axis=1)), smooth(s45_df.max(axis=1))
    s85_mean, s85_min, s85_max = smooth(s85_df.mean(axis=1)), smooth(s85_df.min(axis=1)), smooth(s85_df.max(axis=1))

    # Anchor SSP lines (mean + envelope) to the historical endpoint - no visual gap.
    anchor_date = h_mean.index[-1]

    def anchor(mean_s, lo_s, hi_s):
        m = pd.concat([pd.Series([h_mean.iloc[-1]], index=[anchor_date]), mean_s])
        lo = pd.concat([pd.Series([h_min.iloc[-1]], index=[anchor_date]), lo_s])
        hi = pd.concat([pd.Series([h_max.iloc[-1]], index=[anchor_date]), hi_s])
        return m, lo, hi

    s45_mean, s45_min, s45_max = anchor(s45_mean, s45_min, s45_max)
    s85_mean, s85_min, s85_max = anchor(s85_mean, s85_min, s85_max)

    def to_records(mean_s, lo_s, hi_s, gcm_df):
        records = []
        for date in mean_s.index:
            rec = {"date": date.strftime("%Y-%m-%d"), "mean": round(float(mean_s[date]), 4),
                   "min": round(float(lo_s[date]), 4), "max": round(float(hi_s[date]), 4)}
            if date in gcm_df.index:
                for g in gcms:
                    v = gcm_df.loc[date, g]
                    rec[g] = None if pd.isna(v) else round(float(v), 4)
            records.append(rec)
        return records

    return {
        "variable": variable.value,
        "granularity": granularity.value,
        "gcms_included": gcms,
        "historical": to_records(h_mean, h_min, h_max, h_df),
        "projected_ssp45": to_records(s45_mean, s45_min, s45_max, s45_df),
        "projected_ssp85": to_records(s85_mean, s85_min, s85_max, s85_df),
    }
