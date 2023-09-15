#!/bin/bash
# Usage
# git clone https://github.com/surcyf123/dataset_enrichment && cd dataset_enrichment && git checkout pierre/download_and_host_gptq && cd .. && bash ~/dataset_enrichment/enrichment_pipeline/prod/prod_onstart.sh && bash ~/dataset_enrichment/enrichment_pipeline/prod/download_and_start_models.sh 3090 models8x3 && source ~/.bashrc
# Stop the script if any command fails
set -e

# Installation steps
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
source $HOME/miniconda/etc/profile.d/conda.sh
conda create -y -n bit2 python=3.10
conda init
source ~/.bashrc
conda activate bit2
pip3 install --upgrade pip
pip3 install --upgrade Pillow
pip3 install flask nvitop tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum exllamav2
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
npm install -g pm2
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash