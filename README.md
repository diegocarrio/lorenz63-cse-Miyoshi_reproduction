# Lorenz-63 CSE: Miyoshi et al. reproduction

Research code to reproduce the Lorenz-63 Control Simulation Experiment (CSE)
with full-state data assimilation and simultaneous control of the `x`, `y`, and
`z` variables. The final notebook reproduces the parameter grid reported in
Figure 3(a) and compares the resulting success ratios with the values
transcribed from the paper.

## Scope

This first public release contains only the full-state experiment:

- all three Lorenz-63 variables are observed and assimilated;
- the CSE perturbation is applied to all three variables (`x`, `y`, and `z`);
- regime changes are selected using the Miyoshi/Qiwen event-selection rule;
- the control horizon `T/T0` and perturbation magnitude `D` are swept over the
  Figure 3(a) grid;
- each experiment uses 40 events and deterministic seeds (`1000 + event index`).

The `z`-only experiments are intentionally outside the scope of this release.

## Requirements

- Python 3.10 or newer
- JupyterLab or Jupyter Notebook
- Enough disk space for the generated long nature run and assimilation
  histories. These binary files can occupy several gigabytes.

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Run Jupyter from the repository root so that the notebooks can import the local
modules and exchange generated files through relative paths.

## Reproduction workflow

Execute the notebooks in this order:

1. `01_generate_nature.ipynb` generates the Lorenz-63 nature run.
2. `02_generate_observations.ipynb` generates synthetic observations of
   `x`, `y`, and `z`.
3. `03_initialize_assimilation.ipynb` configures the ensemble data-assimilation
   system and writes `x_analysis_init.pkl`.
4. `04_run_osse.ipynb` runs the observing system simulation experiment and
   writes the analysis history and summary used by CSE.
5. `05_reproduce_figure3a_xyz.ipynb` selects the 40 Miyoshi/Qiwen events, runs
   the resumable `T/T0` by `D` grid, and creates the Figure 3(a) comparison.

The complete grid is computationally expensive. Every `(T, D)` result is
checkpointed after each event, so an interrupted run can be resumed by
executing the same cell again.

## Main configuration

The final notebook explicitly sets:

```python
CONTROL_INDICES = None       # perturb x, y, and z
REGIME_REFERENCE_MODE = "nature"
EVENT_SELECTION = "qiwen"
N_BATCH_EVENTS = 40
```

The Figure 3(a) grid uses:

- `T/T0`: 1.5, 2.0, 2.5, 3.0, and 4.0
- `D`: 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, and 0.5

## Generated files

Pickles, NumPy checkpoints, CSV tables, and figures are not versioned. They are
generated locally and excluded through `.gitignore`. This keeps the repository
small while preserving the complete code path needed to recreate them.

## Reproducibility notes

- Random seeds are fixed wherever the experimental workflow uses stochastic
  initialization or perturbations.
- The notebooks have been cleared of execution outputs and personal filesystem
  paths.
- The numerical modules retain their original research-code structure to keep
  the reproduction aligned with the experiment that produced the reported
  results.

## Reference

Miyoshi, T. and Sun, Q. (2022), “Control simulation experiment with Lorenz's
butterfly attractor”, *Nonlinear Processes in Geophysics*, 29, 133–139.
https://doi.org/10.5194/npg-29-133-2022
