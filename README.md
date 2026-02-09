## versions

### old versions (for performance comparison)
- ver1_numpy: uses numpy and vectorisation
- ver2_numba: uses numba to accelerate
- ver3_numba_streaming: add streaming to lower RAM cost

### new versions (recommended)
- ver4_online_idx: calculate z_index online (when needed) and enable parallelisation
- ver5_cuda: uses numba-cuda to enable GPU acceleration


## enviroment setup
1. create conda environment and activate it (assuming it's called `optics`)
2. install dependencies
```sh
conda install -c nvidia cuda-toolkit=12.6 -y
conda install -c conda-forge numba numpy scipy tifffile -y
```


## how to run (choose one)
1. run construciton*.py directly: run `python construction*.py` for usages
2. use runner: run `python run.py -h` for usages


## NOTE
- Seems like I carelessly named all volumes as "frame"s. = =;