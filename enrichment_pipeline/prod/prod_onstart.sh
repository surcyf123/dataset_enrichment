#!/bin/bash

# Stop the script if any command fails
set -e

# Installation steps
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
source $HOME/miniconda/etc/profile.d/conda.sh
conda create -y -n myenv python=3.10
conda activate myenv
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
npm install -g pm2
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <gpu_type> <model_choices>"
    echo "gpu_type should be either '3090' or '4090'."
    echo "model_choices should be one of 'models4x1', 'models4x2', 'models8x1', or 'models8x2'."
    exit 1
fi

gpu_type=$1
model_choice=$2

# Validate GPU type
if [[ "$gpu_type" != "3090" && "$gpu_type" != "4090" ]]; then
    echo "Invalid gpu_type. Choose either '3090' or '4090'."
    exit 1
fi

# Define model arrays
declare -A model_arrays
model_arrays["models4x1"]=("Luban-13B-GPTQ" "Mythical-Destroyer-V2-L2-13B-GPTQ" "Speechless-Llama2-13B-GPTQ" "Stheno-Inverted-L2-13B-GPTQ")
model_arrays["models4x2"]=("StableBeluga-13B-GPTQ" "CodeUp-Llama-2-13B-Chat-HF-GPTQ" "Llama2-13B-MegaCode2-OASST-GPTQ" "Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GPTQ")
model_arrays["models8x1"]=("UndiMix-v2-13B-GPTQ" "OpenOrca-Platypus2-13B-GPTQ" "orca_mini_v3_13B-GPTQ" "PuddleJumper-13B-GPTQ" "Chronoboros-Grad-L2-13B-GPTQ" "Firefly-Llama2-13B-v1.2-GPTQ" "Airolima-Chronos-Grad-L2-13B-GPTQ" "YuLan-Chat-2-13B-GPTQ")
model_arrays["models8x2"]=("LosslessMegaCoder-Llama2-13B-Mini-GPTQ" "Mythical-Destroyer-V2-L2-13B-GPTQ" "LoKuS-13B-GPTQ" "Luban-13B-GPTQ" "Huginn-13B-v4-GPTQ" "Huginn-13B-v4.5-GPTQ" "Huginn-v3-13B-GPTQ" "Stheno-Inverted-L2-13B-GPTQ")

# Check if the chosen model array exists
if [[ -z "${model_arrays["$model_choice"]}" ]]; then
    echo "Invalid model_choices. Choose one of 'models4x1', 'models4x2', 'models8x1', or 'models8x2'."
    exit 1
fi

models=("${model_arrays["$model_choice"][@]}")

# Change to the home directory
cd ~/

# Clone each model repository
for model in "${models[@]}"; do
    git clone "https://huggingface.co/TheBloke/$model"
done

cd ~/dataset_enrichment/enrichment_pipeline/production_deployment
# Define port range
start_port=30000
end_port=$((start_port + ${#models[@]} - 1))
ports=($(seq $start_port $end_port))

# Ensure that the number of models and ports are the same
if [[ ${#models[@]} -ne ${#ports[@]} ]]; then
    echo "Error: The number of models and ports must be the same."
    exit 1
fi

# Get public IP
ip=$(curl -s https://ifconfig.me)

# Initialize an empty array for URLs
declare -a urls

# Start PM2 processes
for i in "${!models[@]}"; do
    model="~/${models[$i]}"  # The model path is relative to the parent directory
    port=${ports[$i]}
    gpu_id=$i  # This assumes you start with GPU ID 0 and increase by 1 for each model

    # Append the URL to the urls array
    urls+=("http://$ip:$port")

    echo "Starting model $model on port $port with GPU ID $gpu_id..."
    pm2 start host_gptq.py --name "${models[$i]}" --interpreter python3 -- "$model" "$port" "$gpu_id"
done

# Print the list of URLs as a Python list
echo "All models started!"
echo -n "["
for i in "${!urls[@]}"; do
    if [[ $i -ne 0 ]]; then
        echo -n ","
    fi
    echo -n "\"${urls[$i]}\""
done
echo "]"