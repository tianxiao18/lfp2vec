# LFP2Vec: Self-Supervised Representation Learning for Local Field Potentials

## Overview

*LFP2Vec* is a self-supervised learning framework for extracting representations from Local Field Potential (LFP) signals using a wav2vec2-inspired architecture. This repository provides the implementation used in our experiments on multiple public LFP datasets.

## Setup

Create the environment using:

```bash
conda create -n lfp2vec -c conda-forge --override-channels python=3.10
conda activate lfp2vec
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt` pins the package versions used to produce the paper results.

## Data

All three datasets are public. The download and preprocessing scripts for each dataset are in:

* `script/allen_preprocessing/` — Allen Brain Observatory Neuropixels (downloaded through `allensdk`)
* `script/ibl_preprocessing/` — IBL Brainwide Map (downloaded through `ONE-api`, see `downloader.py`)
* `script/Neuronexus_preprocessing/` — Buzsaki Lab Neuronexus recordings

The preprocessing scripts write LFP data and corresponding labels as `.pkl` files in the following structure:

```
data/
├── Allen/
    ├── lfp/
    ├── raw/
    └── spectrogram/
├── ibl/
    ├── lfp/
    ├── raw/
    └── spectrogram/
└── Neuronexus/
    ├── lfp/
    ├── raw/
    └── spectrogram/
```


## Running the Model

The main script for training and evaluation is:

```bash
python script/wav2vec_random_init.py
```

### Key Arguments

* `--data`: Dataset to use (`Allen`, `ibl`, `Neuronexus`)
* `--data_type`: Type of data (`raw`, `lfp`, `spectrogram_preprocessed`, etc.)
* `--ssl`: Enable self-supervised training (`True`/`False`)
* `--rand_init`: Use random initialization instead of pretrained models

Example:

```bash
python script/wav2vec_random_init.py --data Allen --data_type spectrogram_preprocessed --ssl True
```

## License

This project is licensed under the MIT License.