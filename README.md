# BERT-Based-GitHub-Issue-Classification
This repository contains source code for the paper titled *BERT-Based GitHub Issue Report Classification*. The project submitted for *The first edition of the NLBSE’22 tool competition* which is on automatic issue report classification, an essential task in issue management and prioritization.
## System information
**Tested on**:

* OS: Red Hat Enterprise Linux
* Kernel: Linux 3.10.0-1160.53.1.el7.x86_64
* Architecture: x86-64

The computing node consists of Dual Twelve-core 2.2GHz Intel Xeon processors - 24 total cores, 128 GB
RAM, and 4 NVIDIA GeForce GTX 1080 Ti GPU accelerator. We used a single core and one GPU for training and evaluation.

**Dependencies**:
* Conda version: 4.10.3
* PyTorch: 1.10.2
* Cuda version: 11.3
## Steps to run the code
0. Clone this repository and w
 ```
 git clone https://github.com/s2e-lab/BERT-Based-GitHub-Issue-Classification.git
 cd BERT-Based-GitHub-Issue-Classification
 ```
1. Run the following script in the specific folders to create the environment.

 ```
 sh script.sh
 ```
2. Run the train.py to train the model. You can check out **train.ipynb** to have an interactive experience.
 ```
 python3 train.py
 ```
 It takes about 18 hours per iteration to train the model on GPU. The code will be exited if you run it on the CPU.
 
3. Run the test.py to evaluate the model. You can check out **test.ipynb** to have an interactive experience.
 ```
 python3 test.py
 ```

## Abstract

