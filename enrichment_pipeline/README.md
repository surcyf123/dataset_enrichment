This package will create flask endpoints of a given model in the configuration file

# Setting Up
The following are instructions on all the requirements to run experiments

### VastAI setup
This assumes that you are working with one of the CUDA images. This is the one I like to use:`pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`

### Installing Requirements
```bash
pip3 install tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ
```

### Installing CUDA and Requirements Yourself (Good Luck)
```bash
conda create -n benchmark python=3.11 anaconda
conda activate benchmark
# conda install -c nvidia cuda-python
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit #-- REPLACE THIS WITH YOUR CUDA VERSION --#
python -m pip install cuda-python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/

pip3 install tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum
```

