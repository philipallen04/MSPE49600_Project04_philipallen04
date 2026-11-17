# MSPE 49600 Project 4: Fluent Simulations and Gaussian Process Modeling

**Student:** Philip Allen  
**Course:** MSPE 49600 – Data Analytics for Motorsports  
**Semester:** Fall 2026

## Study

This project uses a 2D NACA 0012 airfoil as a simplified rear-wing section. A seeded one-dimensional Latin-hypercube design over 0–20° provides ten CAD rotation angles. Fluent drag and lift results are modeled with separate Gaussian Process regressions, validated on three held-out CFD cases, then refit on all ten runs to quantify predictive mean and variance. The engineering objective is to identify a useful drag-to-downforce region and recommend one follow-up Fluent simulation.

## Updated data note

The updated `airfoil_results(2).xlsx` supplied for this rebuild contained **9 of the 10 seeded LHS runs**. The missing seeded angle was then supplied separately from the completed Fluent results in the same `(angle_deg, drag_N, lift_N)` format:

- angle = **6.738699054°**
- drag = **23.182230 N**
- lift = **-1116.603600 N**

Those measured values are used as the tenth CFD observation. No force value is interpolated or model-generated. The repository `airfoil_results.csv` is the resulting complete ten-run source table used by the notebook.

## Validation

After sorting by angle, the 3rd, 6th, and 9th runs are held out for testing, as required by the assignment.

- Drag test RMSE: **31.127057 N**
- Lift test RMSE: **106.449846 N**
- Largest absolute drag error: **53.899 N at 17.764°**
- Largest absolute lift error: **182.812 N at 17.764°**

The high-angle region is therefore treated with more caution than the mid-angle region.

## Final Gaussian Process models

After validation, both models are refit using all ten CFD points.

- Drag GP: approximately `0.85² × RBF(length_scale=2.24°)`
- Lift GP: approximately `1.35² × RBF(length_scale=2.86°)`

The largest predictive variances occur near the low-angle gap around 1.7–1.8°, not near the final recommended candidate.

## Engineering recommendation

A **500 N minimum predicted downforce** is used before evaluating drag/downforce ratio.

The recommended follow-up Fluent simulation is:

**9.87°**

GP prediction at the candidate:

- drag ≈ **32.09 N**
- lift ≈ **-1563.67 N**
- downforce ≈ **1563.67 N**
- drag/downforce ≈ **0.020521**
- drag predictive variance ≈ **1.35 N²**
- lift predictive variance ≈ **2.71 N²**

The best sampled case is 9.61°, with about 31.67 N drag, 1535.79 N downforce, and an observed ratio of about 0.020618. The predicted 9.87° candidate is an interior point and should be verified with one additional Fluent run before being treated as a physical optimum.

## Files

- `run_airfoil_sweep.py` – corrected seeded 10-angle LHS sweep
- `run_airfoil_project.py` – Workbench automation helper
- `airfoil_profile.py` – NACA 0012 coordinate generator
- `NACA0012_300_DesignModeler.txt` – DesignModeler coordinate file
- `airfoil_results.csv` – complete ten-run force table used by the notebook
- `philipallen04_pr04_cfd.csv` – sorted CFD data and drag/downforce ratio
- `philipallen04_pr04_notebook.ipynb` – executed Project 4 analysis
- `philipallen04_pr04_notebook.py` – Python export for reproducibility
- `requirements.txt` – Python package list
- `.gitignore` – excludes virtual environments and ANSYS/Fluent case-data files

## Python packages

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- jupyter / ipykernel
- ansys-workbench-core (only required for Workbench automation)

## Rerunning the analysis

### Notebook analysis only

1. Place `airfoil_results.csv` and `philipallen04_pr04_notebook.ipynb` in the same folder.
2. Install the Python packages in `requirements.txt`.
3. Run the notebook from top to bottom.

The notebook regenerates `philipallen04_pr04_cfd.csv` and all required tables/plots.

### Full Fluent sweep

1. Open the Workbench project and start the server with `StartServer()`.
2. Update `PORT`, `SECURITY`, and the parameter IDs in `run_airfoil_project.py` if required.
3. Activate the project virtual environment.
4. Run `run_airfoil_sweep.py`.

The virtual environment and Fluent case/data files should remain outside the public repository.

## Engineering limitation

This is a 2D section study. It does not represent finite-span effects, tip vortices, endplates, vehicle-body interaction, ground interaction, yaw, or nonuniform inflow. The recommended angle is therefore a follow-up CFD candidate, not a final race-car setup decision.