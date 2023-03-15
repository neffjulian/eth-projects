# CIL 2022

CIL 2022 course project

## Installation

These instructions assume that Python is installed. The required python version for this project is Python 3.10.

**1. Install requirements**

```sh
python -m pip install -r requirements.txt
```

> Note: The provided requirements assume that cuda is available on the system. If you want to run it on a cpu, consider installing the cpu version of pytorch


### Dataset
The dataset is available through Kaggle. Downloading the dataset from Kaggle requires valid credentials installed on the system. Please follow the instructions to add credentials for the [Kaggle API](https://github.com/Kaggle/kaggle-api). If you want to manually download the data, please save it according to the folder structure provided below. Manual creation of small images and Massachusetts dataset is not possible.

```sh
cd src
python make_dataset.py
```

If the credentials are set correctly, the dataset is downloaded and extracted to the data folder.

The massachusetts dataset can be created by running the following script

```sh
cd src
python make_dataset_massachusetts.py
```

## Run
To run an experiment, simply pass the desired configuration file as command line argument to the main.py script

```sh
cd src
python main.py -c configs/deeplabv3plus_timm-efficientnet-b3.yaml
```

> Note: To run it on a cpu, please remove the `accelerator` and `devices` property from the trainer configuration

## Project Organization

    ├── LICENSE
    ├── Makefile              <- Makefile with commands like `make data` or `make train`
    ├── README.md             <- The top-level README for developers using this project.
    │
    ├── configs               <- Collection of configuration files for experiments
    │
    ├── data
    │   ├── processed         <- The final, canonical data sets for modeling.
    │   └── raw               <- The original, immutable data dump.
    │
    ├── notebooks              <- Jupyter notebooks.
    │
    ├── report                 <- Final report
    │   └── figures            <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                    <- Source code for use in this project.
    │   ├── __init__.py        <- Makes src a Python module
    │   ├── baseline_unet.py   <- (old) UNet implementation
    │   ├── dataset.py         <- Collection of functions and classes for data loading
    │   ├── make_dataset.py    <- Generator script for dataset
    │   ├── UNet.py            <- Model implementation
    │   └── utils.py           <- Helper functions and classes
    │
    └── main.py                <- Main script to run experiments
    
## Used Models

We provide checkpoints of the models we used in https://polybox.ethz.ch/index.php/s/WyJ6tidrkkB4mBm

## Authors

- [@Julian Neff](https://github.com/neffjulian)
- [@Colin Arnet](https://github.com/carnet98)
- [@Remo Kellenberger](https://github.com/remo48)
