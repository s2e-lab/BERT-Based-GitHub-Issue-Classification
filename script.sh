# To store the database
mkdir Dataset 
# To store the model states
mkdir Models
# Download train set and move dataset folder.
curl "https://tickettagger.blob.core.windows.net/datasets/github-labels-top3-803k-train.tar.gz" | tar -xz 
mv github-labels-top3-803k-train.csv ./Dataset/
# Download test set and move dataset folder.
curl "https://tickettagger.blob.core.windows.net/datasets/github-labels-top3-803k-test.tar.gz" | tar -xz 
mv github-labels-top3-803k-train.csv ./Dataset/
# Create a conda environment with Python 3, Tested with Python 3.9.7
conda create -n nlbse python=3 
# Activate the environment
conda activate nlbse
# Install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# Install tqdm and transformers
conda install tqdm transformers 
# Required for using transformets
pip install tensorflow