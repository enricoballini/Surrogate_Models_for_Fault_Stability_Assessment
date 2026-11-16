# Surrogate Models for Fault Stability Assessment

Benchmark for surrogate models that predict the stability of a fault during the
production of a gas reservoir and the following injection of CO2.

Geological CO2 storage changes the pressure of the reservoir and, as a consequence,
the stress acting on the nearby faults. The Coulomb Failure Function (CFF) is the shear
stress on the fault in excess of its frictional resistance: it indicates how close the
fault is to slip. Computing it requires a flow simulation coupled with a solid
mechanics simulation, which is too expensive to repeat for many scenarios. The task is
to learn a fast surrogate of the map

    (9 parameters, time)  ->  CFF on the 416 cells of the fault

from a small number of simulated scenarios.

## Data

The test case is a modified version of the PUNQ-S3 reservoir with six wells and a
planar fault, described in the references below. The schedule has 16 years of gas
production, 20 years of shut-in and 10 years of CO2 injection.

| Input        | Description                                           | Range               |
|--------------|-------------------------------------------------------|---------------------|
| `WGIR_PRO-*` | CO2 injection rate of each of the six wells           | 0 to 1e5 sm3/day    |
| `Cm`         | reservoir compressibility, as in the simulation input | 2.63e-4 to 3.225e-4 |
| `k0`         | initial horizontal/vertical effective stress ratio    | 0.3 to 0.7          |
| `theta`      | fault dip angle                                       | 90 to 120 degrees   |
| time         | years since 1 January 1968                            | 0 to 45.5           |

The output is the CFF in MPa on the 416 triangular cells of the fault mesh.

The data are in HDF5 files in the `data` folder:

| File            | Scenarios | Times | Notes                                         |
|-----------------|-----------|-------|-----------------------------------------------|
| `train.h5`      | 34        | 9     | 11 Plackett-Burman and 23 Latin hypercube     |
| `validation.h5` | 10        | 9     | Latin hypercube                               |
| `test.h5`       | 10        | 11    | Latin hypercube, not public                   |
| `fault_mesh.h5` |           |       | nodes and triangles of the fault, for plots   |

Each data file contains:

- `parameters`: shape (scenarios, 9)
- `times`: shape (times,)
- `cff`: shape (scenarios, times, 416)
- `scenario_ids`: identifiers of the scenarios in the original simulations

The eleven times are 0, 3, 7, 11, 15.5, 25.5, 35.5, 38.5, 40.5, 43.5 and 45.5 years.
The last two, during the CO2 injection, are only in the test set: they measure how
well a model extrapolates in time. The validation and test scenarios have injection
rates and stress ratio in the upper half of their ranges.

The test set is hidden. Predictions on the test set are evaluated through the Kaggle
challenge at <https://www.kaggle.com/organizations/sciml-geobench/competitions>.

## Run

Create the Python environment, activate it and run all the steps from the main folder
of the repository:

    ./install.sh
    source .venv/bin/activate
    ./run.sh

`install.sh` creates a virtual environment in the `.venv` folder and installs the
packages of `requirements.txt` in it. The code has been tested on Ubuntu 26.04.1 LTS
(Linux kernel 7.0.0, x86_64) with Python 3.14.4.

`run.sh` trains the reference model, saves its predictions, computes the metrics and
makes the figures. Everything is written in the `results` folder. The figures are
optional: they are not part of the assessment, they only help to interpret the results.

To run a single script, set `PYTHONPATH` to the main folder first:

    export PYTHONPATH=$PWD
    python3 model_and_training/train.py

## Test your own model

The folder `model_and_training` contains the reference model, a deep learning reduced
order model (DL-ROM) written with PyTorch. Replace it with your model and its training.

Read the data with PyTorch:

    from src.data.torch_dataset import FaultStabilityDataset
    training_data = FaultStabilityDataset("train")
    inputs, cff = training_data[0]

or with JAX:

    import jax
    from src.data.jax_dataset import load_jax_samples, minibatches
    inputs, cff = load_jax_samples("train")
    for batch_inputs, batch_cff in minibatches(inputs, cff, 8, jax.random.key(0)):
        ...

or as NumPy arrays:

    from src.data.dataset import load_samples
    inputs, cff = load_samples("train")

Each sample has 10 inputs, the 9 parameters followed by the time, and the 416 values
of the CFF.

Save the predictions of your model for the inputs of a split:

    from src.data.dataset import load_samples, save_predictions
    inputs, _ = load_samples("validation")
    save_predictions("validation", my_model(inputs))

Then compute the metrics, which do not depend on the model:

    python3 src/metrics/compute_metrics.py

Optionally, make the figures. They are not part of the assessment, they only help to
interpret the results:

    python3 src/plots/plot_rmse_boxplot.py
    python3 src/plots/plot_rmse_in_time.py
    python3 src/plots/plot_cff_on_fault.py

## Metrics

Accuracy is measured with the RMSE of DCFF = CFF_predicted - CFF_reference, in MPa. It
is computed for each split, separately for the nine times of the training set and for
the two future times. A physically acceptable error is about 0.1 MPa.

Efficiency is measured with the number of trainable parameters and with the inference
time per scenario. Write them in `results/efficiency.json` as in
`model_and_training/predict.py` to have them reported with the other metrics.

All the metrics are saved in `results/metrics.json`. The reference DL-ROM gives:

| RMSE of DCFF [MPa]         | Reference DL-ROM |
|----------------------------|------------------|
| validation                 | 0.32             |
| test, nine training times  | 0.24             |
| test, two future times     | 0.22             |

with 22263 trainable parameters and an inference time of about 0.1 ms per scenario
on a CPU. The errors change by several hundredths of MPa with the random seed.

## Folders

    data/                 HDF5 files of the benchmark
    model_and_training/   reference model (DL-ROM), its training and its predictions
    src/data/             data reading for NumPy, PyTorch and JAX users
    src/metrics/          metrics of the predictions
    src/plots/            optional figures
    install.sh            creates the Python environment
    run.sh                runs everything

`src/data/extract_data.py` creates the HDF5 files from the original simulation files,
which are not public.

## References

- E. Ballini et al., *Enhancing computational efficiency of numerical simulation for
  subsurface fluid-induced deformation using deep learning reduced order models*,
  Computational Geosciences 30 (2026). <https://doi.org/10.1007/s10596-026-10409-6>
- A. Forello et al., *A Functional Surrogate Modelling Approach for Fault Stability
  Studies*, Fifth EAGE Conference on Petroleum Geostatistics (2023).
  <https://doi.org/10.3997/2214-4609.202335009>

## License

The code is released under the MIT License, see `LICENSE`.

The data in the `data` folder are released under the Creative Commons Attribution 4.0
International License (CC BY 4.0), see `data/LICENSE`. If you use them, cite the
references above.

Code and data are provided as they are, without warranty of any kind.
