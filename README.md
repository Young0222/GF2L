# GF2L
Graph Frequency Filtering Learning (GF2L)

Graph Frequency Filtering Learning (GF2L)
==================================================

This repository contains the implementation of GF2L (Graph Frequency Filtering Learning), as proposed in our paper:

Beyond Contrast and Generation: Graph Frequency Filtering Learning for Self-Supervised Graph Neural Networks [Anonymous authors] Submitted to WWW 2026

Method Overview
----------------
GF2L is a self-supervised graph learning framework that operates in the frequency domain. It decomposes graph signals into low-, mid-, and high-frequency components, and integrates them through frequency correlation, reconstruction, and adaptive optimization.

Dependencies
-------------
Install required packages using:

    pip install -r requirements.txt

How to Run
-----------
We provide scripts for running GF2L on different learning tasks:

1. Unsupervised Learning: bash run_us_gf2l.sh

2. Transfer Learning: bash run_ts_gf2l.sh

3. Semi-Supervised Learning: cd semi_supervised; bash run_ss_gf2l.sh

Log Files
----------
Training logs and evaluation outputs are saved in the 'log_file' directory. You can refer to these for example runs and performance tracking.

Pretrained Models
------------------
In transfer learning tasks, the pretrained GF2L model files are stored in:

    ./models_gf2l/chem/*.pth
    
Dataset Availability
---------------------
Due to the large size of some datasets, we will make the full dataset package publicly available in a subsequent release. We have provided partial experimental datasets in the folder of original_datasets.
