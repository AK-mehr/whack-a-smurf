# SAML-D strategic launderer experiment

Notebook: `SAML-D.ipynb`. Plan: `preregistration_draft.md`.

## What you need

- Python **3.11 or newer** (pandas 3.x does not install on older versions).
- About 16 GB of RAM is recommended (not measured precisely: the raw CSV is about 1 GB, and the first run builds large temporary tables).
- An internet connection on the first run. The notebook installs its own packages and downloads the dataset (see below), so there is nothing to set up beforehand.

## Run

1. Open `SAML-D.ipynb` in VS Code (with the Python and Jupyter extensions) or in Jupyter, and select a Python 3.11+ interpreter as the kernel. If VS Code offers to install `ipykernel`, accept.
2. Run the cells from the top, in order.

The first cells take care of the rest:
- **Install packages** installs numpy, pandas, pyarrow and matplotlib at the tested versions, only if they are missing or different. If it says to restart the kernel, do that and run again from the top.
- **Get the dataset** downloads `SAML-D.csv` (about 1 GB, from a 200 MB archive) from Kaggle if it is not next to the notebook. No Kaggle account is needed. To get it by hand instead, download it from https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml and put `SAML-D.csv` in the same folder as the notebook.

The first full run also builds a cache (`SAML-D.parquet` and `account_ids.npy`). Later runs load the cache and are much faster. If you change the preprocessing, delete both cache files so they are rebuilt.

A virtual environment is optional but keeps the packages out of your main Python. Create one with `python -m venv saml`, then select it as the kernel.

## Troubleshooting

- **Package installation failed**: no internet, a proxy, or `pip` missing in the selected environment. Install by hand with `pip install numpy==2.4.6 pandas==3.1.0 pyarrow==26.0.0 matplotlib==3.11.2`.
- **`Download failed`**: usually a firewall or proxy. Download the CSV by hand (link above). Python's downloader uses the `HTTPS_PROXY` environment variable if one is set.
- **`FileNotFoundError: SAML-D.csv`**: the CSV is missing or the notebook is not running from this folder.
- **Install fails with a path-too-long error (Windows)**: move the project closer to the drive root (for example `C:\saml`) or enable long paths (https://pip.pypa.io/warnings/enable-long-paths).
- **Out of memory**: close other programs, and in the setup cell reduce `WINDOWS = [7, 14, 30]` to `[30]`.
- **Different results**: check that the installed versions match the install cell (`pip list`).
