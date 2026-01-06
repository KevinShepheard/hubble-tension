# Entropy-Guided Reanalysis of the Hubble Tension

This repository contains the **minimal, paper-facing analysis code** used to generate the figures and quantitative claims in the manuscript:

**Disentangling Calibration Bias and Ordered Structure in Type Ia Supernova Residuals:  
An Entropy-Guided Reanalysis of the Hubble Tension**  
Kevin Shepheard (2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18164054.svg)](https://doi.org/10.5281/zenodo.18164054)

The purpose of this code is not to provide a general cosmological inference framework, but to **cleanly, conservatively, and transparently reproduce** the specific empirical results reported in the paper.

---

## Scope and intent

This repository is designed to answer one narrow question:

> Are ordered residual structure and distance-scale calibration bias separable effects in SN Ia data?

Accordingly, the code:
- Is intentionally small and auditable
- Avoids model refitting or hidden degrees of freedom
- Produces only the artifacts required to support the paper’s claims

---

## What the code does

The analysis pipeline:

- Loads the Pantheon+SH0ES Type Ia supernova dataset
- Constructs a smooth baseline distance–redshift relation μ_ref(z)
- Computes Hubble residuals relative to this baseline
- Partitions the data into redshift × host-mass cells
- Computes, per cell:
  - Intercept shifts (calibration-sensitive)
  - Ordering-dependent entropy statistics
- Applies global conditioning on:
  - Host-galaxy stellar mass
  - Light-curve stretch (x1)
  - Color (c)
- Produces three figures corresponding one-to-one with Results sections in the paper:
  1. Calibration sensitivity (intercept shift)
  2. Persistence of ordered structure (entropy)
  3. Orthogonality of the two effects

All outputs are written to a single, timestamped run directory.

---

## What the code does *not* do

- It does not fit cosmological models
- It does not refit the Hubble diagram after conditioning
- It does not infer a new value of H₀
- It does not propose or test new physics

The analysis is strictly empirical and diagnostic.

---

## Requirements

- Python 3.9 or newer
- NumPy
- Pandas
- Matplotlib

No cosmology-specific libraries are required.

---

## Running the analysis

Example invocation:

python3 paper_proof.py \
  --pantheon data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat \
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

This produces a directory containing:
- manifest.json (full configuration)
- report.json (compact reviewer-facing summary)
- baseline_cells.csv
- candidate_HOST_LOGMASS_cells.csv
- candidate_x1_cells.csv
- candidate_c_cells.csv
- Three figure PNGs used directly in the manuscript

---

## Reproducibility notes

- All random processes use fixed seeds
- Cell binning is deterministic (equal-count)
- The baseline μ_ref(z) is never refit after conditioning
- Entropy statistics are explicitly ordering-dependent and treated as such

These design choices are intentional and documented in both the code and the paper.

---

## License

This repository is released under the **MIT License**.

You are free to use, modify, and redistribute the code with attribution.  
No warranty is provided.

---

## Citation

If you use or reference this code, please cite the associated paper (Zenodo DOI forthcoming):

Shepheard, K. (2026). *Disentangling Calibration Bias and Ordered Structure in Type Ia Supernova Residuals.*

---

## Contact

Questions, critiques, and replication attempts are welcome via GitHub issues.
