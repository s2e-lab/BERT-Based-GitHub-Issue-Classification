# BERT-Based-GitHub-Issue-Classification

## System information
Tested on:
Red Hat Enterprise Linux, Kernel: Linux 3.10.0-1160.53.1.el7.x86_64, Architecture: x86-64

Conda version: conda 4.10.3

## Steps to run the code
1. Run the following commands in the specific folders to create the environment

    ```
    # To store the database
    mkdir Dataset 
    # To store the model states
    mkdir Models
    # Create a conda environment with Python 3, Tested with Python 3.9.7
    conda create -n nlbse python=3 
    # Activate the environment
    conda activate nlbse
    # Install pytorch
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    # Install tqdm and transformers
    conda install tqdm transformers 
    # Required for using transformets
    pip install tensorflow # Required for using transformets
    ```
2. Run the train.py to train the model.
3. Run the test.py to evaluate the model.
