"""
tools.py
--------
All data-reading tools for the Well Log & Seismic Q&A Agent.

Every public function here is a plain Python function that can be unit-tested
independently. The LangChain StructuredTool wrappers at the bottom of the file
turn them into tools the LangGraph agent can call.
"""

from __future__ import annotations

import os
import glob
import re
from typing import Optional

import numpy as np
import pandas as pd
import lasio
import segyio
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ===========================================================================
# ── 1. list_wells ──────────────────────────────────────────────────────────
# ===========================================================================

def list_wells(data_dir: str) -> list[dict]:
    """
    Scan *data_dir*/wells/ and return one dict per LAS file containing:
      - well_name : value of the WELL header field (or filename stem if absent)
      - filename  : absolute path to the LAS file
    """
    wells_dir = os.path.join(data_dir, "wells")
    if not os.path.isdir(wells_dir):
        raise FileNotFoundError(f"Wells directory not found: {wells_dir}")

    results: list[dict] = []
    for las_path in sorted(glob.glob(os.path.join(wells_dir, "*.las"))
                           + glob.glob(os.path.join(wells_dir, "*.LAS"))):
        try:
            las = lasio.read(las_path)
            # lasio stores well info in las.well; try common field names
            well_name = None
            for key in ("WELL", "WN", "NAME"):
                try:
                    val = las.well[key].value
                    if val and str(val).strip():
                        well_name = str(val).strip()
                        break
                except Exception:
                    pass
            if not well_name:
                well_name = os.path.splitext(os.path.basename(las_path))[0]
        except Exception as exc:
            well_name = os.path.splitext(os.path.basename(las_path))[0]

        results.append({"well_name": well_name, "filename": las_path})

    return results


# ===========================================================================
# ── 2. list_seismic_surveys ────────────────────────────────────────────────
# ===========================================================================

def list_seismic_surveys(data_dir: str) -> list[dict]:
    """
    Scan *data_dir*/seismic/ and return one dict per SEG-Y file containing:
      - survey_name : extracted from textual header (line 2 preferred) or filename
      - filename    : absolute path to the SEG-Y file
    """
    seismic_dir = os.path.join(data_dir, "seismic")
    if not os.path.isdir(seismic_dir):
        raise FileNotFoundError(f"Seismic directory not found: {seismic_dir}")

    patterns = ["*.segy", "*.SEGY", "*.sgy", "*.SGY"]
    segy_paths: list[str] = []
    for pat in patterns:
        segy_paths.extend(glob.glob(os.path.join(seismic_dir, pat)))
    segy_paths = sorted(set(segy_paths))

    results: list[dict] = []
    for segy_path in segy_paths:
        survey_name = _extract_survey_name(segy_path)
        results.append({"survey_name": survey_name, "filename": segy_path})

    return results


def _extract_survey_name(segy_path: str) -> str:
    """Try to read the survey name from the SEG-Y textual header (line 2)."""
    try:
        with segyio.open(segy_path, ignore_geometry=True) as f:
            raw = f.text[0]
            # segyio returns bytes or str depending on version
            if isinstance(raw, (bytes, bytearray)):
                text = raw.decode("ascii", errors="replace")
            else:
                text = str(raw)
            # Split into 80-char lines
            lines = [text[i: i + 80].strip() for i in range(0, len(text), 80)]
            for line in lines:
                # Look for "SURVEY NAME:" or "SURVEY:" pattern
                m = re.search(r"SURVEY\s*(?:NAME)?\s*[:\-]\s*(.+)", line, re.IGNORECASE)
                if m:
                    name = m.group(1).strip().rstrip("C").strip()
                    if name:
                        return name
    except Exception:
        pass
    # Fallback: use filename stem
    return os.path.splitext(os.path.basename(segy_path))[0]


# ===========================================================================
# ── 3. list_available_curves ───────────────────────────────────────────────
# ===========================================================================

def list_available_curves(well_file: str) -> list[str]:
    """
    Return a list of curve mnemonic names present in ``well_file``.

    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(well_file):
        raise FileNotFoundError(f"Well file not found: {well_file}")

    las = lasio.read(well_file)
    return [curve.mnemonic for curve in las.curves]


# ===========================================================================
# ── 4. get_curve_stats ─────────────────────────────────────────────────────
# ===========================================================================

def get_curve_stats(well_file: str, curve_name: str,
                    depth_min: float, depth_max: float) -> dict:
    """
    Return ``{"curve": ..., "depth_min": ..., "depth_max": ...,
              "count": ..., "mean": ..., "min": ..., "max": ...}``
    for *curve_name* inside [depth_min, depth_max].

    Raises:
        FileNotFoundError   – if the file doesn't exist
        ValueError          – if the curve is absent, depth range is invalid,
                              or no valid samples exist in the range
    """
    if not os.path.isfile(well_file):
        raise FileNotFoundError(f"Well file not found: {well_file}")
    if depth_min >= depth_max:
        raise ValueError(
            f"depth_min ({depth_min}) must be less than depth_max ({depth_max})."
        )

    las = lasio.read(well_file)
    available = [c.mnemonic.upper() for c in las.curves]
    curve_upper = curve_name.upper()
    if curve_upper not in available:
        raise ValueError(
            f"Curve '{curve_name}' not found in {os.path.basename(well_file)}. "
            f"Available curves: {', '.join(available)}"
        )

    df = las.df().reset_index()
    depth_col = df.columns[0]                        # first column is DEPT/DEPTH
    df = df.rename(columns={depth_col: "DEPT"})
    df.columns = [c.upper() for c in df.columns]

    mask = (df["DEPT"] >= depth_min) & (df["DEPT"] <= depth_max)
    subset = df.loc[mask, curve_upper].replace(las.well.get("NULL", -999.25).value
                                               if hasattr(las.well.get("NULL", None), "value")
                                               else -999.25, np.nan)
    subset = subset.dropna()

    if subset.empty:
        raise ValueError(
            f"No valid (non-null) samples for curve '{curve_name}' "
            f"between {depth_min} m and {depth_max} m."
        )

    return {
        "well_file": os.path.basename(well_file),
        "curve": curve_name.upper(),
        "depth_min_m": depth_min,
        "depth_max_m": depth_max,
        "count": int(subset.count()),
        "mean": round(float(subset.mean()), 4),
        "min": round(float(subset.min()), 4),
        "max": round(float(subset.max()), 4),
        "unit": _get_curve_unit(las, curve_name),
    }


def _get_curve_unit(las: lasio.LASFile, curve_name: str) -> str:
    for c in las.curves:
        if c.mnemonic.upper() == curve_name.upper():
            return c.unit or "unknown"
    return "unknown"


# ===========================================================================
# ── 5. flag_well_anomalies ─────────────────────────────────────────────────
# ===========================================================================

def flag_well_anomalies(well_file: str, curve_name: str,
                        threshold: float, above: bool = True) -> list[dict]:
    """
    Return depth intervals where *curve_name* is above (or below) *threshold*.

    Each returned dict has keys: ``depth_start``, ``depth_end``, ``sample_count``.

    Raises ValueError / FileNotFoundError as appropriate.
    """
    if not os.path.isfile(well_file):
        raise FileNotFoundError(f"Well file not found: {well_file}")

    las = lasio.read(well_file)
    available = [c.mnemonic.upper() for c in las.curves]
    curve_upper = curve_name.upper()
    if curve_upper not in available:
        raise ValueError(
            f"Curve '{curve_name}' not found. Available: {', '.join(available)}"
        )

    df = las.df().reset_index()
    depth_col = df.columns[0]
    df = df.rename(columns={depth_col: "DEPT"})
    df.columns = [c.upper() for c in df.columns]

    null_val = -999.25
    series = df[curve_upper].replace(null_val, np.nan)
    depths = df["DEPT"].values
    values = series.values

    direction = "above" if above else "below"
    condition = (values > threshold) if above else (values < threshold)
    # Also exclude NaN
    condition = condition & ~np.isnan(values)

    intervals: list[dict] = []
    in_interval = False
    start_depth: float = 0.0
    count = 0

    for i, (d, flag) in enumerate(zip(depths, condition)):
        if flag and not in_interval:
            in_interval = True
            start_depth = float(d)
            count = 1
        elif flag and in_interval:
            count += 1
        elif not flag and in_interval:
            intervals.append({
                "depth_start_m": start_depth,
                "depth_end_m": float(depths[i - 1]),
                "sample_count": count,
                "condition": f"{curve_upper} {direction} {threshold}",
            })
            in_interval = False
            count = 0

    if in_interval:
        intervals.append({
            "depth_start_m": start_depth,
            "depth_end_m": float(depths[-1]),
            "sample_count": count,
            "condition": f"{curve_upper} {direction} {threshold}",
        })

    return intervals


# ===========================================================================
# ── 6. get_seismic_survey_info ─────────────────────────────────────────────
# ===========================================================================

def get_seismic_survey_info(seismic_file: str) -> dict:
    """
    Return geometry metadata for a SEG-Y file:
      inline_range, crossline_range, n_samples, sample_interval_ms
    """
    if not os.path.isfile(seismic_file):
        raise FileNotFoundError(f"SEG-Y file not found: {seismic_file}")

    with segyio.open(seismic_file, ignore_geometry=True) as f:
        all_inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
        all_crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

        dt_us = segyio.tools.dt(f)          # sample interval in microseconds
        dt_ms = dt_us / 1000.0
        n_samples = f.bin[segyio.BinField.Samples]
        if n_samples == 0:
            n_samples = len(f.samples)

    return {
        "seismic_file": os.path.basename(seismic_file),
        "inline_min": int(all_inlines.min()),
        "inline_max": int(all_inlines.max()),
        "crossline_min": int(all_crosslines.min()),
        "crossline_max": int(all_crosslines.max()),
        "n_traces": int(len(all_inlines)),
        "n_samples_per_trace": int(n_samples),
        "sample_interval_ms": round(float(dt_ms), 3),
        "total_time_ms": round(float(dt_ms) * n_samples, 1),
    }


# ===========================================================================
# ── 7. get_seismic_amplitude_stats ─────────────────────────────────────────
# ===========================================================================

def get_seismic_amplitude_stats(seismic_file: str, inline: int, crossline: int,
                                time_min_ms: float, time_max_ms: float) -> dict:
    """
    Return amplitude statistics for the trace at (inline, crossline)
    within [time_min_ms, time_max_ms].

    Raises:
        FileNotFoundError  – if file missing
        ValueError         – if inline/crossline not found or time range invalid
    """
    if not os.path.isfile(seismic_file):
        raise FileNotFoundError(f"SEG-Y file not found: {seismic_file}")
    if time_min_ms >= time_max_ms:
        raise ValueError(
            f"time_min_ms ({time_min_ms}) must be less than time_max_ms ({time_max_ms})."
        )

    with segyio.open(seismic_file, ignore_geometry=True) as f:
        dt_us = segyio.tools.dt(f)
        dt_ms = dt_us / 1000.0
        n_samples = len(f.samples)

        # Build index: (inline, crossline) -> trace index
        all_il = f.attributes(segyio.TraceField.INLINE_3D)[:]
        all_xl = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

        trace_idx = None
        for idx, (il, xl) in enumerate(zip(all_il, all_xl)):
            if int(il) == inline and int(xl) == crossline:
                trace_idx = idx
                break

        if trace_idx is None:
            il_range = f"({int(all_il.min())}–{int(all_il.max())})"
            xl_range = f"({int(all_xl.min())}–{int(all_xl.max())})"
            raise ValueError(
                f"Inline {inline} / Crossline {crossline} not found in "
                f"{os.path.basename(seismic_file)}. "
                f"Valid inline range: {il_range}, crossline range: {xl_range}."
            )

        trace = f.trace[trace_idx]                    # numpy array of float32
        total_time_ms = dt_ms * n_samples

        if time_min_ms < 0 or time_max_ms > total_time_ms:
            raise ValueError(
                f"Time range [{time_min_ms}, {time_max_ms}] ms exceeds survey "
                f"time extent [0, {total_time_ms:.1f}] ms."
            )

        # Convert ms to sample indices
        idx_min = int(np.floor(time_min_ms / dt_ms))
        idx_max = int(np.ceil(time_max_ms / dt_ms))
        idx_max = min(idx_max, n_samples - 1)

        window = trace[idx_min: idx_max + 1]

    return {
        "seismic_file": os.path.basename(seismic_file),
        "inline": inline,
        "crossline": crossline,
        "time_min_ms": time_min_ms,
        "time_max_ms": time_max_ms,
        "sample_count": int(len(window)),
        "mean_amplitude": round(float(np.mean(window)), 4),
        "min_amplitude": round(float(np.min(window)), 4),
        "max_amplitude": round(float(np.max(window)), 4),
        "rms_amplitude": round(float(np.sqrt(np.mean(window**2))), 4),
    }


# ===========================================================================
# ── Pydantic schemas (required for StructuredTool) ─────────────────────────
# ===========================================================================

class ListWellsInput(BaseModel):
    data_dir: str = Field(..., description="Absolute path to the data root directory")

class ListSeismicInput(BaseModel):
    data_dir: str = Field(..., description="Absolute path to the data root directory")

class ListCurvesInput(BaseModel):
    well_file: str = Field(..., description="Absolute path to the LAS file")

class CurveStatsInput(BaseModel):
    well_file: str = Field(..., description="Absolute path to the LAS file")
    curve_name: str = Field(..., description="Curve mnemonic, e.g. GR, NPHI, RHOB, RT")
    depth_min: float = Field(..., description="Top of depth interval (metres)")
    depth_max: float = Field(..., description="Base of depth interval (metres)")

class AnomaliesInput(BaseModel):
    well_file: str = Field(..., description="Absolute path to the LAS file")
    curve_name: str = Field(..., description="Curve mnemonic, e.g. GR, RT")
    threshold: float = Field(..., description="Numeric threshold value")
    above: bool = Field(True, description="True = flag values above threshold; False = below")

class SeismicInfoInput(BaseModel):
    seismic_file: str = Field(..., description="Absolute path to the SEG-Y file")

class SeismicAmpInput(BaseModel):
    seismic_file: str = Field(..., description="Absolute path to the SEG-Y file")
    inline: int = Field(..., description="Inline number")
    crossline: int = Field(..., description="Crossline number")
    time_min_ms: float = Field(..., description="Start of time window in milliseconds")
    time_max_ms: float = Field(..., description="End of time window in milliseconds")


# ===========================================================================
# ── LangChain StructuredTool wrappers ──────────────────────────────────────
# ===========================================================================

list_wells_tool = StructuredTool.from_function(
    func=list_wells,
    name="list_wells",
    description=(
        "Scan the wells folder and return all LAS files with their well names. "
        "Use this first to identify which file corresponds to a well the user mentioned."
    ),
    args_schema=ListWellsInput,
)

list_seismic_surveys_tool = StructuredTool.from_function(
    func=list_seismic_surveys,
    name="list_seismic_surveys",
    description=(
        "Scan the seismic folder and return all SEG-Y files with their survey names. "
        "Use this first to identify which file corresponds to a survey the user mentioned."
    ),
    args_schema=ListSeismicInput,
)

list_available_curves_tool = StructuredTool.from_function(
    func=list_available_curves,
    name="list_available_curves",
    description=(
        "Return all curve names (GR, NPHI, RHOB, RT, DEPT …) available in a LAS file. "
        "Call this when you need to verify which curves exist before querying stats."
    ),
    args_schema=ListCurvesInput,
)

get_curve_stats_tool = StructuredTool.from_function(
    func=get_curve_stats,
    name="get_curve_stats",
    description=(
        "Return mean/min/max statistics for a named curve within a depth range (metres). "
        "Always provide exact depth_min and depth_max values from the user's question."
    ),
    args_schema=CurveStatsInput,
)

flag_well_anomalies_tool = StructuredTool.from_function(
    func=flag_well_anomalies,
    name="flag_well_anomalies",
    description=(
        "Find depth intervals where a well log curve exceeds (or falls below) a threshold. "
        "Returns a list of depth ranges with the number of samples in each anomalous interval."
    ),
    args_schema=AnomaliesInput,
)

get_seismic_survey_info_tool = StructuredTool.from_function(
    func=get_seismic_survey_info,
    name="get_seismic_survey_info",
    description=(
        "Return the geometry of a SEG-Y survey: inline/crossline ranges, "
        "number of samples per trace, and sample interval in milliseconds."
    ),
    args_schema=SeismicInfoInput,
)

get_seismic_amplitude_stats_tool = StructuredTool.from_function(
    func=get_seismic_amplitude_stats,
    name="get_seismic_amplitude_stats",
    description=(
        "Return mean/min/max/RMS amplitude for a single trace (inline + crossline) "
        "within a time window specified in milliseconds."
    ),
    args_schema=SeismicAmpInput,
)


# Convenience list imported by agent.py
ALL_TOOLS = [
    list_wells_tool,
    list_seismic_surveys_tool,
    list_available_curves_tool,
    get_curve_stats_tool,
    flag_well_anomalies_tool,
    get_seismic_survey_info_tool,
    get_seismic_amplitude_stats_tool,
]
