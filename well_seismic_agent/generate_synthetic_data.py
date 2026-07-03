"""
generate_synthetic_data.py
--------------------------
Generates synthetic sample data files for the Well Log & Seismic Q&A Agent:
  - data/wells/Well-Alpha.las
  - data/wells/Well-Beta.las
  - data/seismic/Survey-Apex.segy

Run once before starting the agent:
    python generate_synthetic_data.py
"""

import os
import numpy as np
import segyio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WELLS_DIR = os.path.join(SCRIPT_DIR, "data", "wells")
SEISMIC_DIR = os.path.join(SCRIPT_DIR, "data", "seismic")
os.makedirs(WELLS_DIR, exist_ok=True)
os.makedirs(SEISMIC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: write a minimal but valid LAS 2.0 file
# ---------------------------------------------------------------------------
def write_las(path: str, well_name: str, depth: np.ndarray, gr: np.ndarray,
              nphi: np.ndarray, rhob: np.ndarray, rt: np.ndarray) -> None:
    """Write a LAS 2.0 file manually (no lasio dependency for generation)."""
    null = -999.25
    lines = []

    # ~Version section
    lines += [
        "~VERSION INFORMATION",
        " VERS.                    2.0 : CWLS Log ASCII Standard - VERSION 2.0",
        " WRAP.                     NO : ONE LINE PER DEPTH STEP",
        "",
    ]

    # ~Well section
    lines += [
        "~WELL INFORMATION",
        f" WELL.              {well_name} : WELL NAME",
        " FLD .                FIELD-A : FIELD NAME",
        " LOC .         SEC 12 T5N R6E : LOCATION",
        " CTRY.                    USA : COUNTRY",
        " SRVC.           SYNTH LOGGER : SERVICE COMPANY",
        " DATE.              2024-01-15 : LOG DATE",
        " UWI .       30-045-20130-00-00 : UNIQUE WELL ID",
        "",
    ]

    # ~Curve section
    lines += [
        "~CURVE INFORMATION",
        " DEPT.M                      : DEPTH",
        " GR  .GAPI                   : GAMMA RAY",
        " NPHI.V/V                    : NEUTRON POROSITY",
        " RHOB.G/CC                   : BULK DENSITY",
        " RT  .OHMM                   : TRUE RESISTIVITY",
        "",
    ]

    # ~Parameter section
    lines += [
        "~PARAMETER INFORMATION",
        f" STRT.M            {depth[0]:.2f} : START DEPTH",
        f" STOP.M            {depth[-1]:.2f} : STOP DEPTH",
        " STEP.M                  0.5 : STEP",
        f" NULL.              {null:.2f} : NULL VALUE",
        "",
    ]

    # ~ASCII data
    lines += ["~ASCII LOG DATA"]
    for d, g, n, r, t in zip(depth, gr, nphi, rhob, rt):
        lines.append(f" {d:10.3f} {g:10.3f} {n:10.4f} {r:10.4f} {t:10.3f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [LAS] Written: {path}")


# ---------------------------------------------------------------------------
# Generate Well-Alpha (shale-dominated with a sandy reservoir zone)
# ---------------------------------------------------------------------------
def gen_well_alpha():
    np.random.seed(42)
    depth = np.arange(1500.0, 3001.0, 0.5)   # 1500–3000 m, 0.5-m step
    n = len(depth)

    # Gamma Ray: mostly shale (80–120 GAPI), sandy zone 2000–2500 m (20–50)
    gr = np.random.uniform(80, 120, n)
    sand_mask = (depth >= 2000) & (depth <= 2500)
    gr[sand_mask] = np.random.uniform(20, 50, sand_mask.sum())
    gr += np.random.normal(0, 2, n)            # noise
    gr = np.clip(gr, 5, 150)

    # Neutron Porosity: shale ~0.30, sand ~0.22
    nphi = np.where(sand_mask, np.random.uniform(0.18, 0.28, n),
                    np.random.uniform(0.28, 0.38, n))
    nphi += np.random.normal(0, 0.005, n)
    nphi = np.clip(nphi, 0.05, 0.55)

    # Bulk Density: shale ~2.55, sand ~2.35
    rhob = np.where(sand_mask, np.random.uniform(2.25, 2.45, n),
                    np.random.uniform(2.45, 2.65, n))
    rhob += np.random.normal(0, 0.01, n)
    rhob = np.clip(rhob, 1.8, 2.9)

    # Resistivity: sand/reservoir ~50–150 Ohm·m, shale 2–10
    rt = np.where(sand_mask, np.random.uniform(40, 180, n),
                  np.random.uniform(1.5, 12, n))
    rt += np.random.normal(0, 0.5, n)
    rt = np.clip(rt, 0.5, 500)

    out = os.path.join(WELLS_DIR, "Well-Alpha.las")
    write_las(out, "Well-Alpha", depth, gr, nphi, rhob, rt)


# ---------------------------------------------------------------------------
# Generate Well-Beta (carbonate-dominated, different depth range)
# ---------------------------------------------------------------------------
def gen_well_beta():
    np.random.seed(99)
    depth = np.arange(2500.0, 4001.0, 0.5)   # 2500–4000 m
    n = len(depth)

    # Gamma Ray: carbonates are typically low GR (15–40 GAPI)
    gr = np.random.uniform(15, 40, n)
    # A thick shale interbedded zone 3000–3200 m
    shale_mask = (depth >= 3000) & (depth <= 3200)
    gr[shale_mask] = np.random.uniform(75, 110, shale_mask.sum())
    gr += np.random.normal(0, 1.5, n)
    gr = np.clip(gr, 5, 150)

    # NPHI: carbonates ~0.10–0.18, shale 0.30–0.40
    nphi = np.where(shale_mask, np.random.uniform(0.28, 0.40, n),
                    np.random.uniform(0.08, 0.20, n))
    nphi += np.random.normal(0, 0.004, n)
    nphi = np.clip(nphi, 0.02, 0.55)

    # RHOB: carbonates ~2.65–2.75, shale 2.40–2.55
    rhob = np.where(shale_mask, np.random.uniform(2.38, 2.55, n),
                    np.random.uniform(2.60, 2.78, n))
    rhob += np.random.normal(0, 0.01, n)
    rhob = np.clip(rhob, 2.0, 2.95)

    # RT: carbonates high (100–800 Ohm·m), shale low (1–8)
    rt = np.where(shale_mask, np.random.uniform(0.8, 8, n),
                  np.random.uniform(80, 900, n))
    rt += np.random.normal(0, 1.0, n)
    rt = np.clip(rt, 0.5, 2000)

    out = os.path.join(WELLS_DIR, "Well-Beta.las")
    write_las(out, "Well-Beta", depth, gr, nphi, rhob, rt)


# ---------------------------------------------------------------------------
# Generate Survey-Apex SEG-Y
# ---------------------------------------------------------------------------
def gen_survey_apex():
    """
    Small 3-D SEG-Y:
      Inlines  : 100–120  (21 inlines)
      Crosslines: 50–70   (21 crosslines)
      Samples  : 500      (at 2 ms dt → 1000 ms max time)
    """
    inlines = np.arange(100, 121)   # 21 inlines
    crosslines = np.arange(50, 71)  # 21 crosslines
    n_samples = 500
    dt_us = 2000                    # sample interval in microseconds (2 ms)

    out = os.path.join(SEISMIC_DIR, "Survey-Apex.segy")

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = 1                 # IBM float
    spec.ilines = inlines
    spec.xlines = crosslines
    spec.samples = np.arange(n_samples, dtype=np.float32)

    np.random.seed(7)

    with segyio.create(out, spec) as f:
        # Textual header (3200 bytes, 40 lines of 80 chars)
        text_header = (
            "C 1 CLIENT: SYNTHETIC OIL CORP                                              "
            "C 2 SURVEY NAME: Survey-Apex                                                "
            "C 3 AREA: NORTH SEA BLOCK 30/6                                              "
            "C 4 SAMPLE INTERVAL (MICROSECONDS): 2000                                    "
            "C 5 SAMPLES PER TRACE: 500                                                  "
            "C 6 INLINES: 100-120   CROSSLINES: 50-70                                    "
            "C 7 GENERATED: 2024-01-15                                                   "
            "C 8                                                                         "
        )
        # Pad to exactly 3200 bytes
        text_header = text_header.ljust(3200)[:3200]
        f.text[0] = text_header

        f.bin.update(
            tsort=segyio.TraceSortingFormat.INLINE_SORTING,
            hdt=dt_us,
            dto=dt_us,
            hns=n_samples,
            mfeet=1,
        )

        # Generate traces: amplitude = chirp + noise, varies by inline
        tr_idx = 0
        for il in inlines:
            for xl in crosslines:
                t = np.linspace(0, 1, n_samples)
                freq = 25 + (il - 100) * 1.5           # Hz-ish variation
                amp_scale = 1000 + (xl - 50) * 50      # amplitude varies by XL
                trace = (amp_scale * np.sin(2 * np.pi * freq * t)
                         * np.exp(-3 * t)
                         + np.random.normal(0, 80, n_samples))

                f.header[tr_idx] = {
                    segyio.TraceField.INLINE_3D: int(il),
                    segyio.TraceField.CROSSLINE_3D: int(xl),
                    segyio.TraceField.TRACE_SEQUENCE_FILE: tr_idx + 1,
                    segyio.TraceField.DelayRecordingTime: 0,
                    segyio.TraceField.SAMPLE_COUNT: n_samples,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
                }
                f.trace[tr_idx] = trace.astype(np.float32)
                tr_idx += 1

    print(f"  [SEG-Y] Written: {out}")
    print(f"          Inlines {inlines[0]}–{inlines[-1]}, "
          f"Crosslines {crosslines[0]}–{crosslines[-1]}, "
          f"{n_samples} samples @ {dt_us//1000} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic data …")
    gen_well_alpha()
    gen_well_beta()
    gen_survey_apex()
    print("\nDone! All sample files created.")
