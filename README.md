# Feature-Based Dataset Fingerprinting for Clustered Federated Learning on Medical Image Data
This repository contains the implementation of our proposed Feature-based dataset FingerPrinting mechanism (FFP) as well as the data loading utilities for the FedMedMNIST LF and LFQ datasets.

The paper including all empirical results can be found on [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2394756)


## Please cite as:
```
@article{scheliga2024feature,
author = {Daniel Scheliga, Patrick Mäder and Marco Seeland},
title = {Feature-Based Dataset Fingerprinting for Clustered Federated Learning on Medical Image Data},
journal = {Applied Artificial Intelligence},
volume = {38},
number = {1},
pages = {2394756},
year = {2024},
publisher = {Taylor \& Francis},
doi = {10.1080/08839514.2024.2394756},
URL = {https://doi.org/10.1080/08839514.2024.2394756},  
}
```

## Abstract:
Federated Learning (FL) allows multiple clients to train a common model without sharing their private training data. In practice, federated optimization struggles with sub-optimal model utility because data is not independent and identically distributed (non-IID). Recent work has proposed to cluster clients according to dataset fingerprints to improve model utility in such situations. These fingerprints aim to capture the key characteristics of clients' local data distributions. Recently, a mechanism was proposed to calculate dataset fingerprints from raw client data. We find that this fingerprinting mechanism comes with substantial time and memory consumption, limiting its practical use to small datasets. Additionally, shared raw data fingerprints can directly leak sensitive visual information, in certain cases even resembling the original client training data. To alleviate these problems, we propose a Feature-based dataset FingerPrinting mechanism (FFP). We use the MedMNIST database to develop a highly realistic case study for FL on medical image data. Compared to existing methods, our proposed FFP reduces the computational overhead of fingerprint calculation while achieving similar model utility.  Furthermore, FFP mitigates the risk of raw data leakage from fingerprints by design.

## Requirements:
You can create a [conda](https://www.anaconda.com/) virtual environment with the following packages:
```
conda create -n FFP python=3.11.3 \
  pytorch=1.13.1 \
  cudatoolkit=11.8 \
  cudnn=8.8.0.121 \
  torchmetrics \
  torchvision \
  torchinfo \
  dill \
  pandas \
  munch \
  matplotlib \
  seaborn \
  pyyaml \
  prettytable
conda activate FFP
pip install fedlab
```
or install it using the provided environment.yaml:
```
conda env create -f environment.yaml
```

## Usage:
We provide three demo notebooks:
+ `FedMedMNIST.ipynb` to load the FedMedMNIST LF and LFQ datasets and illustrate their training data distributions over all clients.
+ `FingerprintingDemo.ipynb` to compute PACFL and FFP dataset fingerprints and visualize the similarity matrices used for client clustering.
+ `PACFLFingerprintLeaks.ipynb` for a demonstration of the potential for direct raw data privacy leakage from PACFL fingerprints.

Furthermore `federated_training.py` can be used to perform Clustered Federated Learning (CFL) with various configurations. 
We provide example configurations in `configs/experiments/`.
These configurations are based on multiple base-configuration files. 
These can be found in `configs/bases/`.
To change specific parameters for the training process, adjust the corresponding base-configuration files.
An optional `--debug` flag can be set for debugging purposes (reduces the amount of communication rounds and epochs of training).
```
python federated_training.py configs/experiments/<config_file>.yaml (--debug)
```


## Credits:
We base our implementation on the following repositories:
+ [1] [GitHub](https://github.com/MMorafah/PACFL) for the implementation of [PACFL](https://ojs.aaai.org/index.php/AAAI/article/view/26197)
+ [2] [GitHub](https://github.com/MedMNIST/MedMNIST) for constructing the FedMedMNIST datasets from various [MedMNIST](https://www.nature.com/articles/s41597-022-01721-8) datasets.