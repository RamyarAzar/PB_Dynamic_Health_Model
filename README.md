# PB Dynamic Health Model (PB-DHM)

**Author:** Ramyar  Azar
**Last updated:** December 6, 2025  

This repository contains a working Python implementation of a PB-based dynamic health model for **a single longitudinal biomarker or risk score**, with an optional **4D “prevention core”** that organizes risk into four latent axes and applies a simple intervention rule.

The code here is the same core used in a private case study and is shared **only as a generic methodological demo**. It does **not** expose any proprietary 7-axis designs or third-party intellectual property. The naming “PB” is used in a generic, abstract sense and is not tied to any external brand or framework.

Alongside this high-level README, the repository includes a more detailed usage and CLI guide (“PB Dynamic Health Model — Version 5”) which documents the script `pb_model_v5.py` and its command-line options.

---

## Snapshot

**Field**  
- Computational health modeling  
- Longitudinal biomarker dynamics & early-drift detection  

**Model type**  
- Single-axis log-normalized state-space model  
- Simple two-phase classifier (“Health” / “EarlyDrift”) based on ratio windows  
- Optional 4D “PB Prevention Core” with a linear potential and gradient-based nudging  

**Stage**  
- Conceptual and numerical prototype  
- Implemented as a lightweight Python script (`pb_model_v5.py`) suitable for small longitudinal series  

**Main question**

Given a small time series of a biomarker or risk score \( y_n \) with time points \( t_n \), can we:

1. Build a **personal baseline** and work in a log-normalized space \( z_n \) that highlights relative drift?  
2. Define **local ratios and smoothed log-drift** that reveal early, subtle departures from baseline (an “early drift” band)?  
3. Classify time steps into **simple phases** (“Health” vs “EarlyDrift”) from these ratios in a transparent, non-black-box way?  
4. Optionally, embed the same series into a **4D latent space** \([I, M, C, B]\) with a simple potential \(V\) and see how a toy intervention rule would push the system back toward a safer basin?

The emphasis is on **clarity and reproducibility** rather than sophistication: the code is small, readable, and designed so that non-programmers can run it via the command line with comma-separated inputs.

---

## Model & approach

### A. Observed series and log-normalized dynamics

The model starts from a single observed time series:

- Time points \( t_0, \dots, t_N \) (e.g., visits in months or years; strictly increasing).  
- Corresponding biomarker or score values \( y_0, \dots, y_N \), all positive.

The user specifies a subset of indices \( H \subset \{0,\dots,N\} \) that clearly correspond to a **healthy baseline** (for example, the first 2–3 visits before any known change). From this, the script computes:

- **Baseline**
  \[
  y_{\text{base}} = \frac{1}{|H|} \sum_{n \in H} y_n
  \]

- **Log-normalized series** (using an estimated log-baseline \( c = \log y_{\text{base}} \)):
  \[
  z_n = \log y_n - c
  \]

- **Step ratios and log-differences**
  \[
  r_n = \frac{y_{n+1}}{y_n}, \quad
  \ell_n = z_{n+1} - z_n
  \]

- **Smoothed log-drift** \(\tilde{\ell}_n\) over a trailing window \(W\), capturing local trend rather than pointwise noise.

- **Early-instability index** \( EI_n \): the fraction of points in the trailing window whose ratio \( r \) lies inside a user-defined “early drift” band \([r_{\text{low}}, r_{\text{high}}]\). This gives a simple 0–1 measure of how persistently the series is exploring an early-risk regime.

The **exit time from the health basin** is defined as the first time \( t_n \) where \(|z_n|\) or \(z_n^2\) exceeds a user-defined threshold. This is a coarse indicator of when the process leaves its personalized normal range.

### B. Simple phase classification (“Health” / “EarlyDrift”)

On top of ratios and smoothed log-drift, the model defines a very simple **two-state phase structure**:

- \(S = \{\text{Health}, \text{EarlyDrift}\}\).  
- For each state, a **canonical ratio** \( r_{\text{center},s} \) and spread \( \sigma_{r,s} \) are specified, defining how “typical” a given ratio is for that state.

Given the smoothed ratio proxy \( r_{\text{smooth},n} \approx \exp(\tilde{\ell}_n) \), the script computes a per-time, per-state **likelihood**:
\[
L_{n,s} \propto \exp\left[-\frac{1}{2} \frac{(r_{\text{smooth},n} - r_{\text{center},s})^2}{\sigma_{r,s}^2} \right].
\]

These are normalized across states to obtain probabilities \( p_{n,s} \), and the **phase index** is simply:
\[
\hat{S}_n = \arg\max_{s \in S} p_{n,s}.
\]

The classifier is intentionally **local and transparent** (no forward–backward HMM yet). The resulting phase index is saved both numerically (in `timeseries.csv`) and visually (as a step plot over time).

### C. Latent log-space dynamics

To demonstrate a minimal dynamic model in log-space, the repository includes a **latent process**:
\[
z_{n+1} = z_n + \mu_{S_n} \Delta t_n + \eta_n,
\]
where:

- \(\mu_{S_n}\) is a state-dependent drift derived from the canonical ratios for each phase,  
- \(\eta_n\) is Gaussian process noise with variance depending on the current phase,  
- \(\Delta t_n = t_{n+1} - t_n\).

This gives a synthetic **latent trajectory** \( z_{\text{sim},n} \) and corresponding **synthetic observed series** \( y_{\text{sim},n} \) through a simple log-observation model. These are useful for sanity checks and for understanding how the phase structure interacts with dynamics, but they are not meant to replace more advanced state-space inference.

### D. Optional 4D PB Prevention Core

As an optional layer, the code can transform the same 1D series into a simple **4D latent vector**:
\[
Z^{(\text{PB})}_n = [I_n, M_n, C_n, B_n]^\top,
\]
where \(I, M, C, B\) are derived from \(y_n\) through basic linear / nonlinear transforms chosen purely for demonstration. (In real use, each axis would be defined from **different** biomarkers or clinical features; here they are placeholders.)

A **linear potential** is defined:
\[
V_n = \alpha_I I_n + \alpha_M M_n + \alpha_C C_n - \beta_B B_n,
\]
and a discrete-time gradient \( \nabla V_n \) is computed from \( V(t_n) \). A toy **intervention rule** then:

- Identifies time steps where \( \nabla V_n > 0 \) (risk increasing),  
- Applies a bounded negative shift to the \(I\)-axis:
  \[
  \Delta I_n = -k \, \nabla V_n
  \]
  clipped by simple constraints and an optional consistency condition,  
- Recomputes \( V_n \) after the intervention.

The result (before / after potential, gradients, and applied ΔZ) is stored in `pb_core.csv` and plotted, giving an **interpretable, sandbox view** of how a simple rule would “push back” against rising risk in a 4D latent space.

Again, this is **generic and illustrative**, not tied to any proprietary architecture and not meant for direct clinical use.

---

## Repository contents

### Top level

- `README.md` — this file (high-level overview, safe for public sharing).  
- `PB Dynamic Health Model — Version 5` — a separate markdown/README file with a detailed CLI & usage guide for `pb_model_v5.py`, including argument descriptions, outputs, and example commands.

### Code

- `pb_model_v5.py` — single-file implementation of the full pipeline:
  - `ObservedSeries` — baseline, log-normalization, ratios, smoothed log-drift, early-instability index, exit time.  
  - `PBPhaseModel` — per-step likelihoods from ratio windows, simple phase classifier, and Markov-chain utilities (for simulation).  
  - `SingleAxisLatentModel` — latent log-space dynamics with phase-dependent drift and noise.  
  - `ObservationModel` — log-space observation model to generate synthetic `y_sim`.  
  - `PBPreventionCore` — optional 4D latent PB axes, potential \(V\), gradient, and intervention rule.  
  - CLI entry point `main()` and the function `run_from_cli(args)` that orchestrates everything.

### Generated outputs

Created under `--outdir` (default `outputs/`):

- `timeseries.csv`  
  - Columns: `t, y, z_obs, r, ell, ell_smooth, EI, phase_idx, z_sim, y_sim`  
- `plot_y.png` — observed biomarker vs time  
- `plot_z.png` — log-normalized observed vs simulated latent trajectory  
- `plot_phase.png` — phase index as a step function  

If `--use_prevention_core` is enabled:

- `pb_core.csv` — potential before/after intervention, gradient, ΔP, and ΔZ components  
- `plot_V.png` — \(V_{\text{before}}(t)\) and \(V_{\text{after}}(t)\)  
- `plot_gradV.png` — gradient of the potential over time  

---

## How to run

For full details and more examples, see the “PB Dynamic Health Model — Version 5” guide in this repo.

### 1. Install dependencies

```bash
pip install numpy matplotlib
```
## 2. Run a minimal example

```bash
python pb_model_v5.py \
  --times  "0,1,2,3,4,5" \
  --values "1.0,0.97,1.02,1.15,1.30,1.45" \
  --healthy "0,1,2" \
  --outdir "outputs_v5_example"
```
### 3. Optional: enable the 4D PB Prevention Core

```bash
python pb_model_v5.py \
  --times  "0,1,2,3,4,5" \
  --values "1.0,0.97,1.02,1.15,1.30,1.45" \
  --healthy "0,1,2" \
  --use_prevention_core \
  --k_gain 0.5 \
  --outdir "outputs_v5_pbcore"
```
Then inspect the CSV files and PNG plots in the chosen output folder.

## Relation to broader work

This repository isolates **only** the concrete PB-based model implemented here:

- A single-axis dynamic health model in log-space  
- Simple, transparent phase classification from ratio windows  
- An optional 4D prevention core as a sandbox for intervention ideas  

It does **not** include or describe any proprietary 7-axis architectures, nor does it expose third-party or collaborator-specific frameworks. Any future extensions (e.g., more axes, richer state-space inference, or integration with other models) can be layered on top of this code in separate, appropriately licensed projects.

## License & usage

This repository is provided as a **research and educational** case study.

- Use is limited to **personal, academic, and non-commercial** purposes.  
- Do **not** use this code as-is in clinical, diagnostic, or commercial products.  
- Redistribution of modified or unmodified versions is **not permitted** without explicit written permission from the author.  

If you are interested in collaboration, extended access, or using this work under a specific agreement, please contact the author directly.
