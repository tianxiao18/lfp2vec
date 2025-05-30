# LFP2Vec: Self-Supervised Representation Learning for Local Field Potentials

## Overview

*LFP2Vec* is a self-supervised learning framework for extracting representations from Local Field Potential (LFP) signals using a wav2vec2-inspired architecture. This repository provides the implementation used in our experiments on multiple public LFP datasets.

## Setup

Create the environment using:

```bash
conda env create -f environment.yml
conda activate lfp2vec
```

## Data

The repository expects LFP data and corresponding labels in the following structure, stored as `.pkl` files. And the preprocessing script is stored in script/dataset_preprocessing/

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