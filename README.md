# BERT-Based-GitHub-Issue-Classification
This repository contains source code for the paper titled *BERT-Based GitHub Issue Report Classification*. The project is accepted for *The first edition of the NLBSE’22 tool competition* which is on automatic issue report classification, an essential task in issue management and prioritization.
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
1. Clone this repository
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
 
 Note: You can download finetuned model from here without running the train.py: https://drive.google.com/drive/folders/1b7my_Hvom0uUP5zxn91jl1UFUNi0cSX8?usp=sharing

3. Run the test.py to evaluate the model. You can check out **test.ipynb** to have an interactive experience.
 ```
 python3 test.py
 ```

## Abstract

Issue tracking is one of the integral parts of software development, especially for open source projects. GitHub, a
commonly used software management tool, provides its own issue tracking system. Each issue can have various tags, which are
manually assigned by the project’s developers. However, manually labeling software reports is a time-consuming and
error-prone task. In this paper, we describe a BERT-based classification technique to automatically label issues as
questions, bugs, or enhancements. Our approach classified reported issues with an F1-score of on average 0.8571 and of with
highest 0.8586. Our technique performs better than the previous FastText based machine learning technique with an F1-score
of 0.8162.

## Cite
1. Citation for the software:
```
 @software{Siddiq_BERT-Based_GitHub_Issue_2022,
 author = {Siddiq, Mohammed Latif and C. S. Santos, Joanna},
 month = {3},
 title = {{BERT-Based GitHub Issue Report Classification}},
 url = {https://github.com/s2e-lab/BERT-Based-GitHub-Issue-Classification},
 version = {1.0.0},
 year = {2022}
 }
```
2. Citation for the paper:
```
@inproceedings{Siddiq_BERT-Based_GitHub_Issue_2022,
  author={Siddiq, Mohammed Latif and C. S. Santos, Joanna},
  title={BERT-Based GitHub Issue Report Classification},
  booktitle={Proceedings of The 1st International Workshop on Natural Language-based Software Engineering (NLBSE'22)},
  doi={10.1145/3528588.3528660},
  year={2022}
}
```
