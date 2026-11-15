"""Convert the original simulation files into the HDF5 files of the benchmark.

Only needed by the maintainers: the original files in data/raw are not public.
It writes data/train.h5, data/validation.h5, data/test.h5 and data/fault_mesh.h5.
"""

import pickle

import h5py
import numpy as np
import pandas as pd

RAW_FOLDER = "data/raw/DM_input_sim_results"
FAULT_FOLDER = "data/raw/modified_PUNQS3_WF/PYTHON_output/CASE_python_0/FS_inputf_poroso_ela"

PARAMETER_COLUMNS = [
    "INJ_PRO-1",
    "INJ_PRO-4",
    "INJ_PRO-5",
    "INJ_PRO-11",
    "INJ_PRO-12",
    "INJ_PRO-15",
    "Cm_din",
    "k0",
    "phi",
]
PARAMETER_NAMES = [
    "WGIR_PRO-1",
    "WGIR_PRO-4",
    "WGIR_PRO-5",
    "WGIR_PRO-11",
    "WGIR_PRO-12",
    "WGIR_PRO-15",
    "Cm",
    "k0",
    "theta",
]
PARAMETER_DESCRIPTIONS = [
    "CO2 injection rate of well PRO-1 [sm3/day]",
    "CO2 injection rate of well PRO-4 [sm3/day]",
    "CO2 injection rate of well PRO-5 [sm3/day]",
    "CO2 injection rate of well PRO-11 [sm3/day]",
    "CO2 injection rate of well PRO-12 [sm3/day]",
    "CO2 injection rate of well PRO-15 [sm3/day]",
    "reservoir compressibility, as in the simulation input",
    "initial horizontal/vertical effective stress ratio [-]",
    "fault dip angle [degrees]",
]

# The simulations have 47 steps: one per year from 1 January 1968 to 1 January 1983,
# then one per year from 1 July 1983 to 1 July 2013. Eleven of them are saved.
SAVED_STEPS = [0, 3, 7, 11, 16, 26, 36, 39, 41, 44, 46]
TIMES = np.array([0, 3, 7, 11, 15.5, 25.5, 35.5, 38.5, 40.5, 43.5, 45.5])
TIME_UNITS = "years since 1 January 1968"

# The last times are kept out of the public files to test the extrapolation in time.
N_FUTURE_TIMES = 2

N_CELLS = 416


def write_split(file_name, scenario_ids, parameters, times, cff):
    with h5py.File(file_name, "w") as file:
        file["scenario_ids"] = scenario_ids
        file["parameters"] = parameters
        file["parameters"].attrs["names"] = PARAMETER_NAMES
        file["parameters"].attrs["description"] = PARAMETER_DESCRIPTIONS
        file["times"] = times
        file["times"].attrs["units"] = TIME_UNITS
        file["cff"] = cff
        file["cff"].attrs["units"] = "MPa"
        file["cff"].attrs["axes"] = "scenario, time, fault cell"
    print(file_name, "-", len(scenario_ids), "scenarios,", len(times), "times")


parameter_table = pd.read_excel(RAW_FOLDER + "/INITIAL_P50_FGIR.xlsx")
parameters = parameter_table[PARAMETER_COLUMNS].to_numpy()

# One column per scenario. The rows are the fault cells, repeated for every time.
cff_table = pd.read_excel(RAW_FOLDER + "/CFF_sim.xlsx", index_col=0)
cff = cff_table.to_numpy().T.reshape(len(parameters), len(TIMES), N_CELLS)
cff = cff / 1e6  # from Pa to MPa

# Scenario 0 is the only one with all the 47 steps: use it to check the saved times.
with open(FAULT_FOLDER + "/tau_sigma_phi.pkl", "rb") as file:
    all_steps_of_scenario_0 = pickle.load(file, encoding="bytes")
cff_all_steps = all_steps_of_scenario_0["CFF"].T / 1e6
assert np.allclose(cff[0], cff_all_steps[SAVED_STEPS])

split = parameter_table["Split"].to_numpy()
training = np.where(split == "T")[0]
validation = np.where(split == "V")[0]
test = np.where(split == "B")[0]

n_public_times = len(TIMES) - N_FUTURE_TIMES
public_times = TIMES[:n_public_times]
public_cff = cff[:, :n_public_times]

write_split("data/train.h5", training, parameters[training], public_times, public_cff[training])
write_split(
    "data/validation.h5", validation, parameters[validation], public_times, public_cff[validation]
)
write_split("data/test.h5", test, parameters[test], TIMES, cff[test])

with open(FAULT_FOLDER + "/Planar_fault_fromEV.pkl", "rb") as file:
    fault = pickle.load(file, encoding="bytes")
with h5py.File("data/fault_mesh.h5", "w") as file:
    file["nodes"] = fault[b"geometry"][b"Node_coordinates"]
    file["nodes"].attrs["units"] = "m"
    file["triangles"] = fault[b"geometry"][b"ID_nodes_element"]
print("data/fault_mesh.h5")
