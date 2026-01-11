"""
simulate_circuits.py

Generates simulated datasets for fault classification in:
- RC
- RL
- RLC circuits

Uses NGSpice via command-line backend (stable on Windows).
"""

import os
import itertools
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# FORCE NGSPICE CLI MODE (CRITICAL FIX)
# ---------------------------------------------------------------------
os.environ["PATH"] += r";C:\ngspice\bin"
os.environ["PYSPICE_USE_SHARED"] = "0"

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.Simulation import NgSpiceShared
# Disable shared DLL completely
NgSpiceShared._shared = None

LOGGER = Logging.setup_logging()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = r"C:\Project\Code"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_CSV = os.path.join(DATA_DIR, "simulations.csv")

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
RC_R_VALUES = [1e3, 4.7e3, 10e3]
RC_C_VALUES = [10e-9, 100e-9, 1e-6]

RL_R_VALUES = [1e3, 4.7e3, 10e3]
RL_L_VALUES = [1e-3, 10e-3]

RLC_R_VALUES = [1e3, 4.7e3]
RLC_L_VALUES = [1e-3, 10e-3]
RLC_C_VALUES = [10e-9, 100e-9]

VIN_AMPLITUDES = [1.0]

u_Ohm: float
u_F: float
u_H: float
u_V: float
u_s: float
u_Hz: float


FAULT_TYPES = [
    "healthy",
    "R_drift", "C_drift", "L_drift",
    "R_open", "C_open", "L_open",
    "R_short", "C_short", "L_short",
]

DRIFT_FACTORS = [0.5, 1.5]

AC_START_FREQ = 10
AC_STOP_FREQ = 1e6
AC_POINTS_PER_DECADE = 40

USE_TRANSIENT = True
TRANSIENT_STOP_TIME = 5e-3
TRANSIENT_STEP_TIME = 1e-5

# ---------------------------------------------------------------------
# Fault helpers
# ---------------------------------------------------------------------
def apply_resistor_fault(circuit, name, value, fault):
    r = circuit.element(name)
    if fault == "R_drift":
        value *= np.random.choice(DRIFT_FACTORS)
    elif fault == "R_open":
        value = 1e12
    elif fault == "R_short":
        value = 1e-3
    r.resistance = value @ u_Ohm
    return value

def apply_capacitor_fault(circuit, name, value, fault):
    c = circuit.element(name)
    if fault == "C_drift":
        value *= np.random.choice(DRIFT_FACTORS)
    elif fault == "C_open":
        value = 1e-18
    elif fault == "C_short":
        value = 1.0
    c.capacitance = value @ u_F
    return value

def apply_inductor_fault(circuit, name, value, fault):
    l = circuit.element(name)
    if fault == "L_drift":
        value *= np.random.choice(DRIFT_FACTORS)
    elif fault == "L_open":
        value = 1e3
    elif fault == "L_short":
        value = 1e-9
    l.inductance = value @ u_H
    return value

# ---------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------
def build_rc(R, C, vin):
    c = Circuit("RC")
    c.SinusoidalVoltageSource("vin", "in", c.gnd, amplitude=vin @ u_V)
    c.R("1", "in", "out", R @ u_Ohm)
    c.C("1", "out", c.gnd, C @ u_F)
    return c

def build_rl(R, L, vin):
    c = Circuit("RL")
    c.SinusoidalVoltageSource("vin", "in", c.gnd, amplitude=vin @ u_V)
    c.R("1", "in", "out", R @ u_Ohm)
    c.L("1", "out", c.gnd, L @ u_H)
    return c

def build_rlc(R, L, C, vin):
    c = Circuit("RLC")
    c.SinusoidalVoltageSource("vin", "in", c.gnd, amplitude=vin @ u_V)
    c.R("1", "in", "n1", R @ u_Ohm)
    c.L("1", "n1", "out", L @ u_H)
    c.C("1", "out", c.gnd, C @ u_F)
    return c

# ---------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------
def run_ac(circuit):
    sim = circuit.simulator()
    return sim.ac(
        start_frequency=AC_START_FREQ @ u_Hz,
        stop_frequency=AC_STOP_FREQ @ u_Hz,
        number_of_points=AC_POINTS_PER_DECADE,
        variation="dec"
    )

def run_transient(circuit):
    sim = circuit.simulator()
    return sim.transient(
        step_time=TRANSIENT_STEP_TIME @ u_s,
        end_time=TRANSIENT_STOP_TIME @ u_s
    )

# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------
def simulate_all():
    rows = []

    # RC
    for R, C, vin in itertools.product(RC_R_VALUES, RC_C_VALUES, VIN_AMPLITUDES):
        for fault in FAULT_TYPES:
            c = build_rc(R, C, vin)
            r = apply_resistor_fault(c, "R1", R, fault)
            cap = apply_capacitor_fault(c, "C1", C, fault)

            ac = run_ac(c)
            for f, v in zip(ac.frequency, ac.out):
                rows.append({
                    "circuit": "RC",
                    "fault": fault,
                    "R": r, "L": np.nan, "C": cap,
                    "freq": float(f),
                    "voltage": float(abs(v))
                })

    return rows

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    LOGGER.info("Starting simulations...")
    rows = simulate_all()
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    LOGGER.info("Saved → %s", OUTPUT_CSV)

if __name__ == "__main__":
    main()
