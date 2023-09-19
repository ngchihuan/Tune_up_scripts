# Tune-up Template 

## 0. Chip Layout, Wiring, and Fabrication 

### 0.1 Chip layout

Put your chip layout here

### 0.2 Wiring

Describe your wiring here

### 0.3 Fabrication Parameters

You might know something about your qubits from fabrication, you can put those here before you start:

|                  | QB1               | QB2          | QB3               | QB4               | QB5          | QB6               |
|------------------|-------------------|--------------|-------------------|-------------------|--------------|-------------------|
| Upper Sweet Spot |  |  |  |  |  |  |
| Lower Sweet Spot |  |  |  |  |  |  |
| Resonator        | 7.0G              | 7.2G         | 7.4G              | 7.1G              | 7.3G         | 7.5G              |

## 1. Imports


```python
# convenience Import for all LabOne Q Functionality
from laboneq.simple import *

# plotting and fitting functionality
from laboneq.contrib.example_helpers.data_analysis.data_analysis import (
    func_invLorentz,
    func_osc,
    fit_Spec,
    fit_Rabi,
    func_decayOsc,
    fit_Ramsey,
)
from laboneq.contrib.example_helpers.plotting.plot_helpers import (
    plot_simulation,
    plot_results,
)

# descriptor imports
from laboneq.contrib.example_helpers.generate_descriptor import generate_descriptor

# for saving results and pulse sheets
from pathlib import Path
import datetime
import time
import scipy
import json
import yaml
from ruamel.yaml import YAML

import matplotlib.pyplot as plt
import numpy as np

from helpers.tuneup_helper import (
    flatten,
    rotate_to_real_axis,
    calc_readout_weight,
    evaluate_rabi,
    evaluate_ramsey,
    evaluate_T1,
    analyze_ACStark,
    analyze_qspec,
    create_x180,
    create_x180_ef,
    create_x90,
    create_x90_ef,
    update_qubit_parameters_and_calibration,
    load_qubit_parameters,
    create_transmon,
    save_results,
)

from helpers.experiment_library import (
    resonator_spectroscopy_parallel_CW_full_range,
    resonator_spectroscopy_single,
    pulsed_resonator_spectroscopy_single,
    qubit_spectroscopy_parallel,
    qubit_spectroscopy_single,
    res_spectroscopy_pulsed_amp_sweep,
    amplitude_rabi_parallel,
    ramsey_parallel,
    t1_parallel,
    ecr_amplitude_sweep,
)

import logging
```

## 2. Set-up


```python
# Initial functions and definiions
ryaml = YAML()
```

### Emulation Mode


```python
emulate = True
```


### Database Set-up


```python
# set up connection to database
demo_setup_db = DataStore("laboneq_data/setup_database.db")

demo_results_db = DataStore("laboneq_data/results_database.db")

# check if data is already stored in database
for key in demo_setup_db.keys():
    print(key)
```

### Device Set-up


```python
test_descriptor = generate_descriptor(
    shfqc_6=["DEV1210X"],
    number_data_qubits=6,
    number_flux_lines=0,
    multiplex=True,
    number_multiplex=6,
    save=True,
    filename="test_descriptor",
)
```


```python
descriptor_file = open("./Descriptors/test_descriptor.yaml").read()
descriptor = ryaml.load(descriptor_file)
```


```python
# define the DeviceSetup from descriptor - additionally include information on the dataserver used to connect to the instruments
demo_setup = DeviceSetup.from_yaml(
    filepath="./Descriptors/test_descriptor.yaml",
    server_host="ip_address",
    server_port="8004",
    setup_name="test",
)
```

### Apply Calibration from File


```python
qubit_parameters = load_qubit_parameters()

qubit_parameters["local_oscillators"]["readout_lo"]["value"] = 7.0e9

transmon_list = update_qubit_parameters_and_calibration(
    qubit_parameters, demo_setup, demo_setup_db
)
# print(demo_setup.get_calibration())
```


```python
transmon_list
```

### Create and Connect to a QCCS Session 

Establishes the connection to the instruments and readies them for experiments


```python
# create and connect to a session
session = Session(device_setup=demo_setup)
session.connect(do_emulation=emulate)
```

## Pulses for Experiments


```python
for transmon in transmon_list:
    print(transmon.parameters.user_defined)
```


```python
def qubit_drive_pulse(qubit):
    return pulse_library.drag(
        uid=f"drag_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"],
        sigma=0.3,
        beta=0.2,
    )
```


```python
def create_amp_sweep(id, start_amp, stop_amp, num_points):
    return LinearSweepParameter(
        uid=f"amp_sweep_{id}",
        start=start_amp,
        stop=stop_amp,
        count=num_points,
    )
```


```python
def readout_gauss_square_pulse(qubit):
    return pulse_library.gaussian_square(
        uid=f"readout_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=qubit.parameters.user_defined["readout_amplitude"],
    )
```


```python
def integration_kernel(qubit):
    return pulse_library.const(
        uid=f"integration_kernel_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=1,
    )
```

## CW Spectroscopy


```python
# define sweep parameter
def create_freq_sweep(
    id, start_freq, stop_freq, num_points, axis_name="Frequency [Hz]"
):
    return LinearSweepParameter(
        uid=f"frequency_sweep_{id}",
        start=start_freq,
        stop=stop_freq,
        count=num_points,
        axis_name=axis_name,
    )
```


```python
cw_spectroscopy_exp = resonator_spectroscopy_parallel_CW_full_range(
    transmon_list[::6],
    create_freq_sweep("outer", 1e9, 8e9, 8),
    create_freq_sweep("inner", -500e6, 500e6, 1001),
)

compiled_cw_spectroscopy_exp = session.compile(cw_spectroscopy_exp)
cw_spectroscopy_results = session.run(compiled_cw_spectroscopy_exp)
```


```python
# access and plot results of one 8GHz sweep
full_data = abs(cw_spectroscopy_results.get_data("resonator_spectroscopy_q0"))

outer = cw_spectroscopy_results.get_axis("resonator_spectroscopy_q0")[0]
inner = cw_spectroscopy_results.get_axis("resonator_spectroscopy_q0")[1]
full_sweep = np.array(flatten([out + inner for out in outer]))

plt.plot(full_sweep, np.array(flatten([data for data in full_data])))
```


```python
save_results(demo_results_db, cw_spectroscopy_results, "cw_spec_results", "full_sweep")
```


```python
single_cw = resonator_spectroscopy_single(
    transmon_list[0],
    create_freq_sweep(f"{transmon_list[0].uid}_sweep", 0, 100e6, 1000),
    measure_range=-10,
    acquire_range=-10,
    # set_lo=True,
    # lo_freq=freq,
)
compiled_single_cw_spect_exp = session.compile(single_cw)
cw_spectroscopy_results = session.run(compiled_single_cw_spect_exp)
```


```python
plot_results(cw_spectroscopy_results)
```


```python
analyze_qspec(res=cw_spectroscopy_results, handle="resonator_spectroscopy_q0")
```


```python
first_readout_res = 7.0e9 + 30430430
```


```python
qubit_parameters["qubits"]["q0"]["readout_resonator_frequency"][
    "value"
] = first_readout_res

transmon_list = update_qubit_parameters_and_calibration(
    qubit_parameters, demo_setup, demo_setup_db
)
```

## Spectroscopy vs Power - "Punchout"


```python
freq_upper = (
    transmon_list[0].parameters.readout_resonator_frequency
    - transmon_list[0].parameters.readout_lo_frequency
    + 50e6
)
freq_lower = (
    transmon_list[0].parameters.readout_resonator_frequency
    - transmon_list[0].parameters.readout_lo_frequency
    - 50e6
)

amp_sweep = SweepParameter(
    uid="amp_sweep2",
    values=np.logspace(start=np.log10(0.001), stop=np.log10(1), num=21),
)

punchout = res_spectroscopy_pulsed_amp_sweep(
    qubit=transmon_list[0],
    integration_kernel=integration_kernel,
    readout_pulse=readout_gauss_square_pulse,
    frequency_sweep=create_freq_sweep(
        f"{transmon_list[0].uid}_sweep", freq_lower, freq_upper, 1001
    ),
    amplitude_sweep=amp_sweep,
    num_averages=2**10,
    measure_range=-25,
    acquire_range=-5,
)

comp_punchout = session.compile(punchout)
punchout_result = session.run(comp_punchout)
```


```python
plot_results(punchout_result)
```


```python
save_results(
    demo_results_db,
    punchout_result,
    "punchout",
    "neg_25_meas_neg_5_acq_range",
)
```


```python
qubit_parameters["multiplex_readout"]["readout_amplitude"]["value"] = 0.55

transmon_list = update_qubit_parameters_and_calibration(
    qubit_parameters, demo_setup, demo_setup_db
)
```

## Continue with your tune-up experiments:

* Qubit Spec
* Rabi
* Ramsey
* T1
* Hahn Echo

What's after that? There are many ways to go once you have done the basics:

* Drag pulse tune-up
* Single shot readout
* Active reset
* f-level tune-up
* Two qubit gates
* Tomography
* RB
