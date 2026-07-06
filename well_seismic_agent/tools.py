"""
tools.py
--------
All data-reading tools for the Well Log & Seismic Q&A Agent.

Every public function here is a plain Python function that can be unit-tested
independently. The LangChain StructuredTool wrappers at the bottom of the file
turn them into tools the LangGraph agent can call.
"""

from __future__ import annotations

import fnmatch
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
# ── 6. find_pay_zones ──────────────────────────────────────────────────────
# ===========================================================================

def _score_interval(mean_vsh: float, mean_phie: float,
                    mean_swe: float, mean_resistivity: float) -> float:
    """Score a pay-zone interval by porosity, saturation, shale content, and resistivity.

    Weights are chosen so usable porosity and hydrocarbon saturation drive the score,
    while shale content and normalized resistivity still contribute without raw resistivity
    overwhelming the result.
    """
    vsh_score = 1.0 - min(mean_vsh / 0.5, 1.0)
    phie_score = min(mean_phie / 0.20, 1.0)
    swe_score = 1.0 - min(mean_swe / 1.0, 1.0)
    resistivity_score = min(mean_resistivity, 2000.0) / 2000.0

    return (
        0.25 * phie_score
        + 0.25 * swe_score
        + 0.15 * vsh_score
        + 0.35 * resistivity_score
    )


def _get_las_depth_step(well_file: str) -> float:
    las = lasio.read(well_file)
    step_value = None
    if hasattr(las.well.get("STEP", None), "value"):
        try:
            step_value = float(las.well.get("STEP").value)
        except Exception:
            step_value = None

    if step_value is not None and step_value > 0:
        return step_value

    df = las.df().reset_index()
    depths = df.iloc[:, 0].astype(float).values
    if len(depths) < 2:
        raise ValueError(f"Unable to determine depth step for {well_file}.")
    return float(np.median(np.diff(depths)))


def _merge_cluster_values(cluster: dict, interval: dict) -> dict:
    total_count = cluster["sample_count"] + interval["sample_count"]
    return {
        "depth_start_m": cluster["depth_start_m"],
        "depth_end_m": interval["depth_end_m"],
        "sample_count": total_count,
        "mean_vsh": round(
            (
                cluster["mean_vsh"] * cluster["sample_count"]
                + interval["mean_vsh"] * interval["sample_count"]
            ) / total_count,
            6,
        ),
        "mean_phie": round(
            (
                cluster["mean_phie"] * cluster["sample_count"]
                + interval["mean_phie"] * interval["sample_count"]
            ) / total_count,
            6,
        ),
        "mean_swe": round(
            (
                cluster["mean_swe"] * cluster["sample_count"]
                + interval["mean_swe"] * interval["sample_count"]
            ) / total_count,
            6,
        ),
        "mean_resistivity": round(
            (
                cluster["mean_resistivity"] * cluster["sample_count"]
                + interval["mean_resistivity"] * interval["sample_count"]
            ) / total_count,
            4,
        ),
        "quality_score": round(
            (
                cluster["quality_score"] * cluster["sample_count"]
                + interval["quality_score"] * interval["sample_count"]
            ) / total_count,
            6,
        ),
    }


def cluster_pay_zones(
    well_file: str,
    max_gap_m: float = None,
    top_n: int = 10,
    **find_pay_zones_kwargs,
) -> dict:
    """
    Group nearby ranked pay-zone intervals into continuous clusters.

    Uses the LAS step interval to default max_gap_m when not provided, and
    returns the top_n clusters by weighted quality score.
    """
    if not os.path.isfile(well_file):
        raise FileNotFoundError(f"Well file not found: {well_file}")

    if max_gap_m is None:
        max_gap_m = 2.0 * _get_las_depth_step(well_file)

    kwargs = dict(find_pay_zones_kwargs)
    kwargs.pop("top_n", None)
    result = find_pay_zones(well_file, top_n=None, **kwargs)
    intervals = result.get("intervals", [])
    if not intervals:
        raise ValueError(
            f"No qualifying pay-zone intervals found in {os.path.basename(well_file)}."
        )

    ordered = sorted(intervals, key=lambda iv: iv["depth_start_m"])
    clusters: list[dict] = []
    current = ordered[0].copy()

    for interval in ordered[1:]:
        if interval["depth_start_m"] - current["depth_end_m"] <= max_gap_m:
            current = _merge_cluster_values(current, interval)
        else:
            current["total_thickness_m"] = round(
                current["depth_end_m"] - current["depth_start_m"], 6
            )
            clusters.append(current)
            current = interval.copy()

    current["total_thickness_m"] = round(
        current["depth_end_m"] - current["depth_start_m"], 6
    )
    clusters.append(current)

    clusters.sort(key=lambda iv: iv["quality_score"], reverse=True)
    total_clusters_found = len(clusters)
    returned_clusters = clusters[:top_n] if top_n is not None else clusters

    return {
        "total_clusters_found": total_clusters_found,
        "clusters": returned_clusters,
    }


def find_best_well_region(
    data_dir: str,
    well_pattern: str = "*",
    top_n: int = 5,
    **cluster_kwargs,
) -> dict:
    """
    Discover the strongest well matching well_pattern and return that well's top clusters.

    This identifies the single best-performing well first, then returns only its top clusters.
    """
    wells = list_wells(data_dir)
    matching = [w for w in wells if fnmatch.fnmatch(w["well_name"], well_pattern)]
    if not matching:
        raise ValueError(
            f"No wells matched pattern '{well_pattern}' in {data_dir}."
        )

    best_well = None
    best_cluster = None
    skipped_wells: list[dict] = []
    wells_compared = 0

    for well in matching:
        well_name = well["well_name"]
        well_file = well["filename"]
        try:
            local_kwargs = dict(cluster_kwargs)
            local_kwargs.pop("top_n", None)
            peak = cluster_pay_zones(well_file, top_n=1, **local_kwargs)
            clusters = peak.get("clusters", [])
            if not clusters:
                raise ValueError(f"No clusters returned for {well_name}.")
            wells_compared += 1
            cluster = clusters[0]
            if best_cluster is None or cluster["quality_score"] > best_cluster["quality_score"]:
                best_cluster = cluster
                best_well = {
                    "well_name": well_name,
                    "well_file": well_file,
                }
        except Exception as exc:
            skipped_wells.append({
                "well_name": well_name,
                "well_file": well_file,
                "reason": str(exc),
            })

    if best_well is None or best_cluster is None:
        raise ValueError(
            f"No valid wells were found for pattern '{well_pattern}'. "
            f"Skipped wells: {[s['well_name'] for s in skipped_wells]}"
        )

    local_kwargs = dict(cluster_kwargs)
    local_kwargs.pop("top_n", None)
    top_clusters_result = cluster_pay_zones(
        best_well["well_file"], top_n=top_n, **local_kwargs
    )

    return {
        "best_well": best_well["well_name"],
        "best_well_file": best_well["well_file"],
        "peak_quality_score": best_cluster["quality_score"],
        "top_clusters": top_clusters_result.get("clusters", []),
        "wells_compared": wells_compared,
        "skipped_wells": skipped_wells,
    }


def find_pay_zones(
    well_file: str,
    vsh_max: float = 0.25,
    phie_min: float = 0.02,
    swe_max: float = 0.8,
    resistivity_min: float = 80.0,
    top_n: int = 10,
) -> dict:
    """
    Return ranked pay zone candidates where VSH, PHIE, SWE, and RESISTIVITY
    meet the provided thresholds.

    Intervals are merged when qualifying depths are within one depth STEP of each other.
    Returns the best scoring intervals first plus the total count of qualifying intervals
    before truncation by top_n.
    """
    if not os.path.isfile(well_file):
        raise FileNotFoundError(f"Well file not found: {well_file}")

    las = lasio.read(well_file)
    available = [c.mnemonic.upper() for c in las.curves]
    required = ["VSH", "PHIE", "SWE", "RESISTIVITY"]
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(
            f"Well file {os.path.basename(well_file)} is missing required curves: "
            f"{', '.join(missing)}"
        )

    df = las.df().reset_index()
    depth_col = df.columns[0]
    df = df.rename(columns={depth_col: "DEPT"})
    df.columns = [c.upper() for c in df.columns]

    null_value = -999.25
    if hasattr(las.well.get("NULL", None), "value"):
        null_value = las.well.get("NULL").value

    for curve in required:
        df[curve] = df[curve].replace(null_value, np.nan)

    # Only accept rows where all required curves are valid
    valid = df[required].notna().all(axis=1)
    condition = (
        (df["VSH"] <= vsh_max)
        & (df["PHIE"] >= phie_min)
        & (df["SWE"] <= swe_max)
        & (df["RESISTIVITY"] >= resistivity_min)
        & valid
    )

    depths = df["DEPT"].values.astype(float)
    if len(depths) < 2:
        return {"total_qualifying_intervals": 0, "intervals": []}

    step = float(np.median(np.diff(depths)))
    max_gap = step + 1e-6

    intervals: list[dict] = []
    indices = np.where(condition)[0]
    if len(indices) == 0:
        return {"total_qualifying_intervals": 0, "intervals": []}

    start_idx = indices[0]
    prev_idx = start_idx

    for idx in indices[1:]:
        if depths[idx] - depths[prev_idx] <= max_gap:
            prev_idx = idx
            continue

        segment = df.loc[start_idx:prev_idx]
        mean_vsh = round(float(segment["VSH"].mean()), 6)
        mean_phie = round(float(segment["PHIE"].mean()), 6)
        mean_swe = round(float(segment["SWE"].mean()), 6)
        mean_resistivity = round(float(segment["RESISTIVITY"].mean()), 4)
        intervals.append({
            "depth_start_m": float(depths[start_idx]),
            "depth_end_m": float(depths[prev_idx]),
            "sample_count": int(len(segment)),
            "mean_vsh": mean_vsh,
            "mean_phie": mean_phie,
            "mean_swe": mean_swe,
            "mean_resistivity": mean_resistivity,
            "quality_score": round(
                _score_interval(mean_vsh, mean_phie, mean_swe, mean_resistivity),
                6,
            ),
        })
        start_idx = idx
        prev_idx = idx

    segment = df.loc[start_idx:prev_idx]
    mean_vsh = round(float(segment["VSH"].mean()), 6)
    mean_phie = round(float(segment["PHIE"].mean()), 6)
    mean_swe = round(float(segment["SWE"].mean()), 6)
    mean_resistivity = round(float(segment["RESISTIVITY"].mean()), 4)
    intervals.append({
        "depth_start_m": float(depths[start_idx]),
        "depth_end_m": float(depths[prev_idx]),
        "sample_count": int(len(segment)),
        "mean_vsh": mean_vsh,
        "mean_phie": mean_phie,
        "mean_swe": mean_swe,
        "mean_resistivity": mean_resistivity,
        "quality_score": round(
            _score_interval(mean_vsh, mean_phie, mean_swe, mean_resistivity),
            6,
        ),
    })

    intervals.sort(key=lambda iv: iv["quality_score"], reverse=True)
    total_qualifying = len(intervals)
    returned_intervals = intervals[:top_n] if top_n is not None else intervals

    return {
        "total_qualifying_intervals": total_qualifying,
        "intervals": returned_intervals,
    }


# ===========================================================================
# ── 7. get_seismic_survey_info ─────────────────────────────────────────────
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

class FindPayZonesInput(BaseModel):
    well_file: str = Field(..., description="Absolute path to the LAS file")
    vsh_max: float = Field(0.25, description="Maximum shale fraction allowed")
    phie_min: float = Field(0.02, description="Minimum effective porosity")
    swe_max: float = Field(0.8, description="Maximum water saturation")
    resistivity_min: float = Field(80.0, description="Minimum resistivity in Ohm·m")
    top_n: int = Field(10, description="Maximum number of ranked candidate intervals to return")

class ClusterPayZonesInput(BaseModel):
    well_file: str = Field(..., description="Absolute path to the LAS file")
    max_gap_m: float = Field(None, description="Maximum allowed gap between intervals to merge into a cluster")
    top_n: int = Field(10, description="Maximum number of top clusters to return")
    vsh_max: float = Field(0.25, description="Maximum shale fraction allowed")
    phie_min: float = Field(0.02, description="Minimum effective porosity")
    swe_max: float = Field(0.8, description="Maximum water saturation")
    resistivity_min: float = Field(80.0, description="Minimum resistivity in Ohm·m")

class FindBestWellRegionInput(BaseModel):
    data_dir: str = Field(..., description="Absolute path to the data root directory")
    well_pattern: str = Field("*", description="Filename pattern to match well names, e.g. 'Z-0*'")
    top_n: int = Field(5, description="Number of top clusters to return for the best well")
    vsh_max: float = Field(0.25, description="Maximum shale fraction allowed")
    phie_min: float = Field(0.02, description="Minimum effective porosity")
    swe_max: float = Field(0.8, description="Maximum water saturation")
    resistivity_min: float = Field(80.0, description="Minimum resistivity in Ohm·m")

import io
import base64
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for server use
import matplotlib.pyplot as plt


def generate_log_plot(well_file: str, vsh_max: float = 0.25,
                       phie_min: float = 0.02, swe_max: float = 0.8,
                       resistivity_min: float = 80.0) -> str:
    """
    Generate a 3-track well log plot (GR/Vsh, Resistivity, PHIE/Swe)
    with the best pay zone interval shaded, returned as a base64 PNG string.

    Raises:
        FileNotFoundError – if the file doesn't exist
        ValueError        – if required curves are missing
    """
    if not os.path.isfile(well_file):
        raise FileNotFoundError(f"Well file not found: {well_file}")

    las = lasio.read(well_file)
    available = [c.mnemonic.upper() for c in las.curves]
    required = ["GR", "VSH", "PHIE", "SWE", "RESISTIVITY"]
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(
            f"Well file {os.path.basename(well_file)} is missing required curves: "
            f"{', '.join(missing)}"
        )

    df = las.df().reset_index()
    depth_col = df.columns[0]
    df = df.rename(columns={depth_col: "DEPT"})
    df.columns = [c.upper() for c in df.columns]

    null_value = -999.25
    if hasattr(las.well.get("NULL", None), "value"):
        null_value = las.well.get("NULL").value
    for curve in required:
        df[curve] = df[curve].replace(null_value, np.nan)

    pay = find_pay_zones(well_file, vsh_max, phie_min, swe_max, resistivity_min, top_n=1)
    best_zone = pay["intervals"][0] if pay["intervals"] else None

    fig, axes = plt.subplots(1, 3, figsize=(9, 10), sharey=True)
    depth = df["DEPT"]

    # Track 1: GR
    axes[0].plot(df["GR"], depth, color="green", linewidth=0.8)
    axes[0].set_xlabel("GR (API)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()

    # Track 2: Resistivity (log scale)
    axes[1].plot(df["RESISTIVITY"], depth, color="red", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Resistivity (Ohm.m)")

    # Track 3: PHIE / SWE
    axes[2].plot(df["PHIE"], depth, color="blue", linewidth=0.8, label="PHIE")
    axes[2].plot(df["SWE"], depth, color="black", linewidth=0.8, linestyle="--", label="SWE")
    axes[2].set_xlabel("PHIE / SWE (frac)")
    axes[2].legend(fontsize=7, loc="upper right")

    if best_zone:
        for ax in axes:
            ax.axhspan(best_zone["depth_start_m"], best_zone["depth_end_m"],
                       color="orange", alpha=0.25)

    well_name = os.path.basename(well_file).replace(".las", "")
    fig.suptitle(f"Well {well_name} — Log Tracks", fontsize=12)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

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

find_pay_zones_tool = StructuredTool.from_function(
    func=find_pay_zones,
    name="find_pay_zones",
    description=(
        "Identify and rank pay-zone candidate intervals in a LAS well file using VSH, PHIE, SWE, and RESISTIVITY thresholds. "
        "Returns the top_n best-scoring merged depth intervals with average log values and a total qualifying count."
    ),
    args_schema=FindPayZonesInput,
)

cluster_pay_zones_tool = StructuredTool.from_function(
    func=cluster_pay_zones,
    name="cluster_pay_zones",
    description=(
        "Group nearby ranked pay-zone intervals into continuous clusters. "
        "Use this when the user asks for sustained intervals, continuous pay zones, or merged zone estimates."
    ),
    args_schema=ClusterPayZonesInput,
)

find_best_well_region_tool = StructuredTool.from_function(
    func=find_best_well_region,
    name="find_best_well_region",
    description=(
        "Find the single best performing well among matching wells and return that well's top clusters. "
        "Use this for questions like 'best zone across my wells' or 'which region is best and what are its zones'."
    ),
    args_schema=FindBestWellRegionInput,
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
    find_pay_zones_tool,
    cluster_pay_zones_tool,
    find_best_well_region_tool,
]
