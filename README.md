# Task Context Improves Population Codes by Warping Response Geometry in Human Visual Cortex

## About

This repository contains the fMRI, behavioral, and eye-tracking analysis code and figure-generation notebooks accompanying the manuscript.

## Data

The data are hosted separately on OSF because of their size. Download `behdata.rar`, `eyedata.rar`, and `fmridata.rar` from [OSF](REPLACE_WITH_OSF_PROJECT_LINK), then extract them into `data/` as follows:

```text
data/
├── behdata/
├── eyedata/
└── fmridata/
```

Keep these directory names unchanged because the analysis code uses relative paths.

## Python environment

The notebooks were developed with Python 3.11. The recommended setup is:

```bash
conda create -n faceprfattention python=3.11 -y
conda activate faceprfattention
python -m pip install -r requirements.txt
```

Run each notebook from its containing directory so that relative paths resolve correctly.

## Repository structure

### `fmri_analysis/`

- `scripts/select_voxels.ipynb`: voxel selection.
- `scripts/session_agg.ipynb`: precomputes the session-wise results for Figures S9 and S10.
- `scripts/cal_metrices.ipynb`: main fMRI analyses.
- `scripts/pRF_model/`: MATLAB scripts for pRF model fitting.
- `expectedoutput/`: saved analysis results used for plotting the figures.

### `beh_analysis/`

- `behperf_analysis.m`: behavioral analysis using data in `data/behdata/`.
- `behperf_results.mat`: saved behavioral results.

### `eye_analysis/`

- `eyetracking.ipynb`: eye-tracking preprocessing and analysis using data in `data/eyedata/`.
- `eye_results/`: saved eye-tracking results.

### `plot_figures/`

- `scripts/`: notebooks for the main and supplementary figures.
- `figures/`: generated PDF figures.

The figure notebooks read the saved outputs directly from the three analysis directories above.

## Additional software

MATLAB is required for the behavioral and pRF analyses. The pRF scripts also require analyzePRF and the MATLAB Parallel Computing Toolbox. Converting raw EyeLink `.edf` files requires SR Research EDF2ASC on the system `PATH`.

## Citation

If you use this repository, please cite:

You, Y.-Q., Li, S., Cheng, Y.-A., Li, Y., Kay, K., & Zhang, R.-Y. (2025). *Attention Improves Population Codes by Warping Neural Manifolds in Human Visual Cortex*. bioRxiv, 2025.2010.2009.681102. https://doi.org/10.1101/2025.10.09.681102

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE) for details.
