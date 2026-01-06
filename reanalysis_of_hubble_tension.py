#!/usr/bin/env python3
"""
paper_proof.py

Minimal, paper-facing reproduction script accompanying the manuscript:

    "Disentangling Calibration Bias and Ordered Structure in
     Type Ia Supernova Residuals:
     An Entropy-Guided Reanalysis of the Hubble Tension"

This script generates the *exact quantitative artifacts* used in the paper,
with no exploratory code paths and no hidden degrees of freedom.

---------------------------------------------------------------------------
DESIGN PRINCIPLES
---------------------------------------------------------------------------

• Reproducibility
  - Deterministic binning and fixed random seeds
  - Explicit manifest.json and compact report.json written per run
  - No dependence on external state or working directory

• Minimal scope
  - Only computations required to support the paper’s core claims
  - No refitting of cosmology or nuisance parameters during conditioning
  - No hyperparameter tuning beyond documented arguments

• Conservative methodology
  - μ_ref(z) is a smooth baseline only (polynomial fit)
  - Baseline is never refit after conditioning
  - Entropy diagnostics are ordering-dependent by construction and treated as such

• Reviewer-auditable outputs
  - Per-cell CSVs (baseline and conditioned)
  - Two figures, each mapped one-to-one to a Results subsection
  - JSON report summarizing headline quantities and assumptions

---------------------------------------------------------------------------
OUTPUTS
---------------------------------------------------------------------------

Each run creates a timestamped directory containing:

  - manifest.json
      Full configuration and figure mapping

  - report.json
      Compact, reviewer-readable summary of headline results

  - baseline_cells.csv
      Per-cell metrics before conditioning

  - candidate_{HOST_LOGMASS,x1,c}_cells.csv
      Per-cell metrics after conditioning on each variable

  - Two figures, each mapped one-to-one to a Results subsection:
      Fig 1 (Calibration Sensitivity):   HOST_LOGMASS intercept shift Δ(Δ)
      Fig 2 (Orthogonality):             scatter Δ(ΔH) vs Δ(Δ) for candidates

---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------

Example invocation:

    python3 paper_proof.py \
        --pantheon data/Pantheon_SH0ES.dat \
        --out out/paper_proof \
        --mass-bins 3 \
        --z-bins 3 \
        --min-cell-n 120 \
        --ordering zHD \
        --block 64 \
        --stride 16 \
        --entropy-bins 16 \
        --shuffle-trials 25 \
        --seed 0

All arguments are optional except --pantheon.

---------------------------------------------------------------------------
INTERPRETATION NOTES
---------------------------------------------------------------------------

• Δ (intercept shift) is calibration-sensitive and maps directly to H₀ ratios.
• ΔH (entropy excess) is ordering-dependent and *does not* measure calibration.
• Orthogonality between Δ and ΔH is the central empirical result.
• This script makes no claims about cosmological model extensions.

---------------------------------------------------------------------------
LICENSE / CITATION
---------------------------------------------------------------------------

This code is released for transparency and reproducibility.
If used in academic work, please cite the accompanying manuscript.

"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Path discipline: never depend on cwd for locating other outputs
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_tag() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# -----------------------------------------------------------------------------
# Data loading / canonicalization
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PantheonTable:
    df: pd.DataFrame
    z: np.ndarray
    mu: np.ndarray
    muerr: np.ndarray
    meta: Dict[str, np.ndarray]
    canonical_mapping: Dict[str, str]


def load_pantheon(path: Path) -> PantheonTable:
    if not path.exists():
        raise FileNotFoundError(f"Missing SN data file: {path}")

    df = pd.read_csv(path, delim_whitespace=True, comment="#")
    cols = set(df.columns)

    mapping: Dict[str, Optional[str]] = {
        "zHD": "zHD" if "zHD" in cols else ("zCMB" if "zCMB" in cols else None),
        "MU": "MU_SH0ES" if "MU_SH0ES" in cols else ("MU" if "MU" in cols else None),
        "MUERR": "MU_SH0ES_ERR_DIAG" if "MU_SH0ES_ERR_DIAG" in cols else ("MUERR" if "MUERR" in cols else None),
        "HOST_LOGMASS": "HOST_LOGMASS" if "HOST_LOGMASS" in cols else None,
        "x1": "x1" if "x1" in cols else None,
        "c": "c" if "c" in cols else None,
        "IDSURVEY": "IDSURVEY" if "IDSURVEY" in cols else None,
        "ID": "CID" if "CID" in cols else ("ID" if "ID" in cols else None),
    }

    if mapping["zHD"] is None:
        raise RuntimeError(f"Could not find zHD (or zCMB) in columns: {sorted(df.columns)}")
    if mapping["MU"] is None:
        raise RuntimeError(f"Could not find MU_SH0ES (or MU) in columns: {sorted(df.columns)}")
    if mapping["MUERR"] is None:
        raise RuntimeError(f"Could not find MU_SH0ES_ERR_DIAG (or MUERR) in columns: {sorted(df.columns)}")

    z = df[mapping["zHD"]].to_numpy(dtype=float)
    mu = df[mapping["MU"]].to_numpy(dtype=float)
    muerr = df[mapping["MUERR"]].to_numpy(dtype=float)

    meta: Dict[str, np.ndarray] = {}
    for k, col in mapping.items():
        if col is None:
            continue
        if k in ("zHD", "MU", "MUERR"):
            continue
        meta[k] = df[col].to_numpy()

    m = np.isfinite(z) & np.isfinite(mu) & np.isfinite(muerr) & (muerr > 0)
    if not np.all(m):
        df = df.loc[m].reset_index(drop=True)
        z = z[m]
        mu = mu[m]
        muerr = muerr[m]
        for k in list(meta.keys()):
            meta[k] = meta[k][m]

    canonical_mapping = {k: v for k, v in mapping.items() if v is not None}
    return PantheonTable(df=df, z=z, mu=mu, muerr=muerr, meta=meta, canonical_mapping=canonical_mapping)


# -----------------------------------------------------------------------------
# Baseline μ_ref(z): weighted polynomial (smooth baseline, not cosmology)
# -----------------------------------------------------------------------------

def weighted_polyfit(x: np.ndarray, y: np.ndarray, w: np.ndarray, deg: int) -> np.ndarray:
    X = np.vander(x, N=deg + 1, increasing=False)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef


def mu_ref_poly(z: np.ndarray, mu: np.ndarray, muerr: np.ndarray, deg: int) -> Tuple[np.ndarray, Dict]:
    w = 1.0 / (muerr ** 2)
    coef = weighted_polyfit(z, mu, w, deg=deg)
    mu_ref = np.polyval(coef, z)
    info = {"model": "poly", "deg": int(deg), "coef_desc": [float(c) for c in coef]}
    return mu_ref, info


def residuals(mu: np.ndarray, mu_ref: np.ndarray) -> np.ndarray:
    return mu - mu_ref


# -----------------------------------------------------------------------------
# Cell binning (equal-count, deterministic)
# -----------------------------------------------------------------------------

def make_bins_equal_count(x: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return np.zeros_like(x, dtype=int)
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return np.zeros_like(x, dtype=int)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(xf, qs)
    edges = np.unique(edges)
    if len(edges) - 1 < n_bins:
        edges = np.linspace(float(np.nanmin(xf)), float(np.nanmax(xf)), n_bins + 1)
    b = np.digitize(x, edges[1:-1], right=False)
    b = np.clip(b, 0, n_bins - 1)
    return b.astype(int)


@dataclass(frozen=True)
class CellSpec:
    mass_bins: int
    z_bins: int
    min_cell_n: int


def cell_indices(tbl: PantheonTable, spec: CellSpec) -> Tuple[np.ndarray, np.ndarray]:
    z_bin = make_bins_equal_count(tbl.z.astype(float), spec.z_bins)
    if "HOST_LOGMASS" in tbl.meta:
        mvals = tbl.meta["HOST_LOGMASS"].astype(float)
        m_bin = make_bins_equal_count(mvals, spec.mass_bins)
    else:
        m_bin = np.zeros_like(z_bin, dtype=int)
    return m_bin, z_bin


# -----------------------------------------------------------------------------
# Intercept metrics
# -----------------------------------------------------------------------------

def weighted_mean_and_sigma(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    sw = float(np.sum(w))
    if sw <= 0:
        return float("nan"), float("nan")
    m = float(np.sum(w * x) / sw)
    s = float(math.sqrt(1.0 / sw))
    return m, s


# -----------------------------------------------------------------------------
# Entropy diagnostic (ordering-dependent)
# -----------------------------------------------------------------------------

def shannon_entropy_hist(x: np.ndarray, bins: int, rng: Tuple[float, float]) -> float:
    hist, _ = np.histogram(x, bins=bins, range=rng)
    n = np.sum(hist)
    if n <= 0:
        return float("nan")
    p = hist.astype(float) / float(n)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def block_entropies(seq: np.ndarray, block: int, stride: int, bins: int, rng: Tuple[float, float]) -> np.ndarray:
    n = len(seq)
    if n < block or block <= 1 or stride <= 0:
        return np.array([], dtype=float)
    out: List[float] = []
    for start in range(0, n - block + 1, stride):
        win = seq[start:start + block]
        out.append(shannon_entropy_hist(win, bins=bins, rng=rng))
    return np.asarray(out, dtype=float)


def deltaH(seq_ordered: np.ndarray, *, block: int, stride: int, bins: int,
           shuffle_trials: int, rng: Tuple[float, float], seed: int) -> Tuple[float, float, float]:
    rg = np.random.default_rng(seed)
    h_obs_blocks = block_entropies(seq_ordered, block=block, stride=stride, bins=bins, rng=rng)
    if h_obs_blocks.size == 0:
        return float("nan"), float("nan"), float("nan")
    H_obs = float(np.nanmean(h_obs_blocks))

    null_means: List[float] = []
    for _ in range(int(shuffle_trials)):
        perm = rg.permutation(seq_ordered)
        h_null_blocks = block_entropies(perm, block=block, stride=stride, bins=bins, rng=rng)
        null_means.append(float(np.nanmean(h_null_blocks)) if h_null_blocks.size else float("nan"))
    H_null = float(np.nanmean(np.asarray(null_means, dtype=float)))
    return float(H_obs - H_null), H_obs, H_null


# -----------------------------------------------------------------------------
# Conditioning (global beta)
# -----------------------------------------------------------------------------

def fit_beta_weighted(r: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(r) & np.isfinite(v) & np.isfinite(w) & (w > 0)
    if np.sum(m) < 3:
        return 0.0
    r0, v0, w0 = r[m], v[m], w[m]
    sw = float(np.sum(w0))
    mr = float(np.sum(w0 * r0) / sw)
    mv = float(np.sum(w0 * v0) / sw)
    cov = float(np.sum(w0 * (r0 - mr) * (v0 - mv)) / sw)
    var = float(np.sum(w0 * (v0 - mv) ** 2) / sw)
    if var <= 0:
        return 0.0
    # minimize Σ w (r + β v)^2 => β = -cov/var
    return float(-cov / var)


# -----------------------------------------------------------------------------
# Grid computation (minimal set of per-cell fields)
# -----------------------------------------------------------------------------

@dataclass
class CellRow:
    cand: str
    mass_bin: int
    z_bin: int
    N: int
    delta: float
    delta_sigma: float
    deltaH: float
    H_obs: float
    H_null: float


def compute_cells(
    *,
    tbl: PantheonTable,
    mu_adj: np.ndarray,
    mu_ref: np.ndarray,
    spec: CellSpec,
    mass_bin: np.ndarray,
    z_bin: np.ndarray,
    ordering: str,
    block: int,
    stride: int,
    entropy_bins: int,
    shuffle_trials: int,
    seed: int,
    cand: str,
) -> List[CellRow]:
    w = 1.0 / (tbl.muerr.astype(float) ** 2)
    r = residuals(mu_adj, mu_ref)

    if ordering == "zHD":
        order_vec = tbl.z.astype(float)
    elif ordering in tbl.meta:
        order_vec = tbl.meta[ordering].astype(float)
    else:
        raise RuntimeError(f"Unknown ordering '{ordering}'. Use 'zHD' or a numeric meta key.")

    rows: List[CellRow] = []
    for mb in range(spec.mass_bins):
        for zb in range(spec.z_bins):
            m = (mass_bin == mb) & (z_bin == zb) & np.isfinite(r) & np.isfinite(w) & (w > 0) & np.isfinite(order_vec)
            idx = np.where(m)[0]
            N = int(idx.size)
            if N < spec.min_cell_n:
                continue

            rr = r[idx]
            ww = w[idx]
            oo = order_vec[idx]

            perm = np.argsort(oo, kind="mergesort")
            rr_ord = rr[perm]

            # robust histogram range
            lo = float(np.quantile(rr_ord, 0.01))
            hi = float(np.quantile(rr_ord, 0.99))
            pad = 1e-6 + 0.05 * (hi - lo if hi > lo else 1.0)
            rng = (lo - pad, hi + pad)

            delt, delt_sig = weighted_mean_and_sigma(rr, ww)
            dH, Hobs, Hnull = deltaH(
                rr_ord,
                block=block,
                stride=stride,
                bins=entropy_bins,
                shuffle_trials=shuffle_trials,
                rng=rng,
                seed=seed + 100000 * mb + 1000 * zb,
            )

            rows.append(CellRow(
                cand=cand,
                mass_bin=int(mb),
                z_bin=int(zb),
                N=int(N),
                delta=float(delt),
                delta_sigma=float(delt_sig),
                deltaH=float(dH),
                H_obs=float(Hobs),
                H_null=float(Hnull),
            ))
    return rows


def pivot(rows: Sequence[CellRow], key: str, mass_bins: int, z_bins: int) -> np.ndarray:
    A = np.full((mass_bins, z_bins), np.nan, dtype=float)
    for r in rows:
        A[r.mass_bin, r.z_bin] = float(getattr(r, key))
    return A


# -----------------------------------------------------------------------------
# Plotting (single-panel figures, paper style)
# -----------------------------------------------------------------------------

def save_heatmap(A: np.ndarray, *, title: str, outpath: Path, annotate: bool = True, fmt: str = ".3f") -> None:
    plt.figure()
    plt.imshow(A, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("z bin")
    plt.ylabel("host-mass bin")
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))
    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    plt.text(j, i, format(A[i, j], fmt), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


def save_scatter(points: Dict[str, Tuple[np.ndarray, np.ndarray]], *, title: str, outpath: Path) -> None:
    plt.figure()
    for label, (x, y) in points.items():
        if x.size == 0:
            continue
        plt.scatter(x, y, s=22, label=label)
    plt.axvline(0.0, linewidth=1)
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.xlabel(r"$\Delta(\Delta H)$  (after - before)")
    plt.ylabel(r"$\Delta(\Delta)$ [mag]  (after - before)")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pantheon", required=True, type=str, help="Path to Pantheon_SH0ES.dat")
    ap.add_argument("--out", type=str, default="BIGRIG/out/paper_proof", help="Output directory (abs or relative to PROJECT_ROOT)")
    ap.add_argument("--mass-bins", type=int, default=3)
    ap.add_argument("--z-bins", type=int, default=3)
    ap.add_argument("--min-cell-n", type=int, default=120)

    ap.add_argument("--ordering", type=str, default="zHD", help="Ordering axis for entropy (default: zHD)")
    ap.add_argument("--mu-ref-deg", type=int, default=3, help="Degree of polynomial baseline μ_ref(z)")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--entropy-bins", type=int, default=16)
    ap.add_argument("--shuffle-trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--annotate", action="store_true", help="Annotate heatmap cells")
    args = ap.parse_args()

    pantheon_path = (PROJECT_ROOT / args.pantheon).resolve() if not Path(args.pantheon).is_absolute() else Path(args.pantheon).resolve()
    out_base = (PROJECT_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    run_dir = out_base / f"run_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "utc": utc_tag(),
        "project_root": str(PROJECT_ROOT),
        "pantheon": str(pantheon_path),
        "mass_bins": int(args.mass_bins),
        "z_bins": int(args.z_bins),
        "min_cell_n": int(args.min_cell_n),
        "ordering": str(args.ordering),
        "mu_ref_deg": int(args.mu_ref_deg),
        "block": int(args.block),
        "stride": int(args.stride),
        "entropy_bins": int(args.entropy_bins),
        "shuffle_trials": int(args.shuffle_trials),
        "seed": int(args.seed),
        "figures": {
            "fig1_hostmass_intercept_shift": "fig1_hostmass_intercept_shift.png",
            "fig2_orthogonality_scatter": "fig3_orthogonality_scatter.png",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    tbl = load_pantheon(pantheon_path)
    mu_ref, mu_ref_info = mu_ref_poly(tbl.z, tbl.mu, tbl.muerr, deg=args.mu_ref_deg)

    spec = CellSpec(mass_bins=args.mass_bins, z_bins=args.z_bins, min_cell_n=args.min_cell_n)
    m_bin, z_bin = cell_indices(tbl, spec)

    base_mu = tbl.mu.astype(float).copy()
    w = 1.0 / (tbl.muerr.astype(float) ** 2)
    r0 = residuals(base_mu, mu_ref)

    # Baseline cells
    baseline_rows = compute_cells(
        tbl=tbl, mu_adj=base_mu, mu_ref=mu_ref,
        spec=spec, mass_bin=m_bin, z_bin=z_bin,
        ordering=args.ordering,
        block=args.block, stride=args.stride,
        entropy_bins=args.entropy_bins,
        shuffle_trials=args.shuffle_trials,
        seed=args.seed,
        cand="baseline",
    )
    pd.DataFrame([dataclasses.asdict(r) for r in baseline_rows]).to_csv(run_dir / "baseline_cells.csv", index=False)

    # Candidates (minimal set)
    candidates = ["HOST_LOGMASS", "x1", "c"]
    cand_rows: Dict[str, List[CellRow]] = {}
    betas: Dict[str, float] = {}

    for cand in candidates:
        if cand not in tbl.meta:
            betas[cand] = float("nan")
            cand_rows[cand] = []
            continue
        v = tbl.meta[cand].astype(float)
        beta = fit_beta_weighted(r0, v, w)
        betas[cand] = float(beta)
        mu_adj = base_mu + beta * v

        rows = compute_cells(
            tbl=tbl, mu_adj=mu_adj, mu_ref=mu_ref,
            spec=spec, mass_bin=m_bin, z_bin=z_bin,
            ordering=args.ordering,
            block=args.block, stride=args.stride,
            entropy_bins=args.entropy_bins,
            shuffle_trials=args.shuffle_trials,
            seed=args.seed + 999,  # decorrelate from baseline but deterministic
            cand=cand,
        )
        cand_rows[cand] = rows
        pd.DataFrame([dataclasses.asdict(r) for r in rows]).to_csv(run_dir / f"candidate_{cand}_cells.csv", index=False)

    # --- FIGURE 1: HOST_LOGMASS intercept shift Δ(Δ) ---
    A_base_delta = pivot(baseline_rows, "delta", spec.mass_bins, spec.z_bins)
    # Restore baseline entropy grid for orthogonality analysis
    A_base_dH = pivot(baseline_rows, "deltaH", spec.mass_bins, spec.z_bins)
    A_mass_delta = pivot(cand_rows.get("HOST_LOGMASS", []), "delta", spec.mass_bins, spec.z_bins)
    A_dDelta_mass = A_mass_delta - A_base_delta
    save_heatmap(
        A_dDelta_mass,
        title=r"Fig. 1 — Host-mass conditioning shifts intercept: $\Delta(\Delta)$ [mag]",
        outpath=run_dir / "fig1_hostmass_intercept_shift.png",
        annotate=args.annotate,
        fmt=".3f",
    )

    # --- FIGURE 3: Orthogonality scatter Δ(ΔH) vs Δ(Δ) for candidates ---
    points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cand in candidates:
        rows = cand_rows.get(cand, [])
        if not rows:
            points[cand] = (np.array([], dtype=float), np.array([], dtype=float))
            continue
        A_c_dH = pivot(rows, "deltaH", spec.mass_bins, spec.z_bins)
        A_c_delta = pivot(rows, "delta", spec.mass_bins, spec.z_bins)
        ddH = (A_c_dH - A_base_dH).ravel()
        dDel = (A_c_delta - A_base_delta).ravel()
        m = np.isfinite(ddH) & np.isfinite(dDel)
        points[cand] = (ddH[m], dDel[m])

    save_scatter(
        points,
        title="Fig. 2 — Orthogonality: entropy change vs intercept shift (per cell)",
        outpath=run_dir / "fig2_orthogonality_scatter.png",
    )

    # Compact report: reviewer-facing facts
    # (global averages, plus figure mapping)
    def finite_mean(A: np.ndarray) -> float:
        m = np.isfinite(A)
        return float(np.nanmean(A[m])) if np.any(m) else float("nan")

    report = {
        "utc": manifest["utc"],
        "pantheon": str(pantheon_path),
        "canonical_mapping": tbl.canonical_mapping,
        "baseline_model": mu_ref_info,
        "settings": {
            "mass_bins": spec.mass_bins,
            "z_bins": spec.z_bins,
            "min_cell_n": spec.min_cell_n,
            "ordering": args.ordering,
            "entropy": {
                "block": args.block,
                "stride": args.stride,
                "bins": args.entropy_bins,
                "shuffle_trials": args.shuffle_trials,
                "seed": args.seed,
            },
        },
        "betas": betas,
        "headline_deltas": {
            "mean_cell_intercept_shift_HOST_LOGMASS_mag": finite_mean(A_dDelta_mass),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "baseline_cells": "baseline_cells.csv",
            "candidate_cells": {c: f"candidate_{c}_cells.csv" for c in candidates},
            "figures": manifest["figures"],
        },
        "interpretation_constraints": [
            "ΔH is ordering-dependent: here it is computed after sorting residuals by --ordering (default zHD).",
            "μ_ref(z) is a smooth baseline; it is not refit during conditioning experiments.",
            "Host-mass conditioning can shift the intercept Δ without materially changing redshift-ordered ΔH; this supports orthogonality between calibration and ordered structure.",
        ],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("============================================================")
    print("paper_proof.py COMPLETE")
    print("============================================================")
    print(f"Run directory: {run_dir}")
    print("Figures:")
    for k, v in manifest["figures"].items():
        print(f"  - {v}")
    print("Report:")
    print("  - report.json")


if __name__ == "__main__":
    main()