# HNCI

## Introduction

The code is for reproducing the numerical experiments and data example results in 'HNCI: High-dimensional Network Causal Inference'.

## Guidelines for Result Replication

### (I) Understanding the Role of Neighborhood Size in Inferring the ADET
  
Please go to folder `./simulation/infer_ADET/simplified_model/` and run `infer-ADET-simplified-model.py`.

### (II) Simulation Results under the General Model

1) **Infer the ADET:** Please go to folder `./simulation/infer_ADET/general_setting/` and run `setting*.py`, where `*` is 1,2,3,4.

2) **Infer the ADET - no exact matching:** Please go to folder `./simulation/infer_ADET/no_exact_matching/` and run `setting*.py`, where `*` is 1,2,3,4.

3) **Infer the ADET - misspecified propensity scores:** Please go to folder `./simulation/infer_ADET/mis_propensity/` and run `setting*.py`, where `*` is 1,2,3,4.

4) **Infer the neighborhood size:** Please go to folder `./simulation/infer_k0/k0_#/`, where `#` is 0,1,2, and run `setting*.py`, where `*` is 1,2,3,4.

### (III) Data Application

Please go to folder `./application_glasgow/` and run `glasgow.py` to obtain the results. The original data files are avalavle <a href="https://www.stats.ox.ac.uk/~snijders/siena/Glasgow_data.htm">here</a>.

## Packed Algorithm Code for Practitioners

`network_utils.py:` Wrapped functions for infer the ADET (`infer_tau`, `sfl_grp`) and neighborhood size (`candidate_set_k_bic`, `confident_set_k`).


## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


