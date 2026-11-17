"""Run after opening the airfoil project in Workbench and typing StartServer()."""

import csv
from functools import cache
from math import isclose, isfinite

from ansys.workbench.core import connect_workbench

CSV_COLUMNS = ["angle_deg", "drag_N", "lift_N"]

# Edit these settings for your project.
PORT = 61572  # Use the port returned by StartServer().
SECURITY = "wnua"  # Local Windows connection.
CAD_ANGLES_DEG = [9.0, 12.0]  # [] reads DP 0. Use your ten LHS angles for the sweep.
CSV_FILE = "airfoil_results.csv"  # Completed runs are appended to this file.
ANGLE_ID, DRAG_ID, LIFT_ID = "P1", "P4", "P5"  # Check your Parameter Set.


@cache
def _connect(port, security):
    return connect_workbench(port=port, host="localhost", security=security)


def _read_or_run(cad_angle_deg):
    # This Python block is sent to Workbench.
    script = f"""
import json

cad_angle_deg = {cad_angle_deg!r}
if cad_angle_deg is None:
    dp = Parameters.GetDesignPoint(Name="0")
else:
    dp = Parameters.CreateDesignPoint()
    angle = Parameters.GetParameter(Name="{ANGLE_ID}")
    dp.SetParameterExpression(Parameter=angle, Expression=str(cad_angle_deg) + " [deg]")
    UpdateAllDesignPoints(DesignPoints=[dp])

values = []
for name in ["{ANGLE_ID}", "{DRAG_ID}", "{LIFT_ID}"]:
    parameter = Parameters.GetParameter(Name=name)
    if not Parameters.IsParameterInDesignPointUpToDate(DesignPoint=dp, Parameter=parameter):
        raise RuntimeError(name + " is not up to date")
    values.append(float(dp.GetParameterValue(Parameter=parameter).Value))

wb_script_result = json.dumps(values)
"""
    result = _connect(PORT, SECURITY).run_script_string(script)
    if result is None:
        raise RuntimeError("Workbench returned no results. See the Workbench error above.")
    return result


def run_airfoil(airfoil_angle_deg):
    """Solve and save one run, then return (drag, signed lift) in N."""
    angle = float(airfoil_angle_deg)
    if not isfinite(angle):
        raise ValueError("The CAD rotation must be a finite angle in degrees.")

    rotation, drag_force_N, lift_force_N = _read_or_run(angle)
    if not isclose(rotation, angle, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(f"Requested {angle:g} deg, but Workbench reported {rotation:g} deg.")
    if not isfinite(drag_force_N) or not isfinite(lift_force_N):
        raise RuntimeError("Workbench returned a nonfinite force.")
    _save_result(angle, drag_force_N, lift_force_N)
    return drag_force_N, lift_force_N


def _save_result(angle_deg, drag_force_N, lift_force_N):
    with open(CSV_FILE, "a+", newline="", encoding="utf-8") as file:
        file.seek(0)
        header = next(csv.reader(file), None)
        if header is not None and header != CSV_COLUMNS:
            raise ValueError(f"Unexpected columns in {CSV_FILE}. Choose a new CSV_FILE.")

        file.seek(0, 2)
        writer = csv.writer(file)
        if header is None:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([angle_deg, drag_force_N, lift_force_N])


if __name__ == "__main__":
    if not CAD_ANGLES_DEG:
        rotation, drag, lift = _read_or_run(None)
        print(f"DP 0: CAD rotation {rotation:g} deg, drag {drag:.4f} N, lift {lift:.4f} N")
    else:
        for angle_deg in CAD_ANGLES_DEG:
            print(f"Running CAD rotation {angle_deg:g} deg...", flush=True)
            drag_force_N, lift_force_N = run_airfoil(angle_deg)
            print(f"Drag {drag_force_N:.4f} N, lift {lift_force_N:.4f} N", flush=True)

        print(f"Results saved to {CSV_FILE}")