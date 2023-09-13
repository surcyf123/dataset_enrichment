#!/bin/bash

# Install Miniconda (you can replace this with Anaconda if preferred)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh

# Initialize Conda
source $HOME/miniconda/etc/profile.d/conda.sh

# Create a Conda environment with Python 3.10
conda create -y -n myenv python=3.10

# Activate the new environment
conda activate myenv

# Now, all commands will run inside the Conda environment
pip install --upgrade Pillow
pip install flask nvitop tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

sed -i "/^PATH='\/opt\/conda\/bin:\/usr\/local\/nvidia\/bin:\/usr\/local\/cuda\/bin:\/usr\/local\/sbin:\/usr\/local\/bin:\/usr\/sbin:\/usr\/bin:\/sbin:\/bin'$/s/^/#/" ~/.bashrc
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
nvm install 14.21.3
nvm alias default 14.21.3
source ~/.bashrc
npm install -g pm2

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Add functionality to call this file with a list of models and it will git clone them all at the end