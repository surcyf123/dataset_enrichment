# Prod Endpoint

### Calling It

```
http://89.177.60.89:40885/process_question
{"prompt" : "proompt", "completions_needed" : 4, "max_time_allowed" : 30,"question_value" : 3}
```

### Installing Deps
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

pip3 install --upgrade Pillow
pip3 install flask tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum

git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip3 install -r requirements.txt
python3 setup.py install --user
```



### Getting the models
```bash
git lfs clone https://huggingface.co/TheBloke/Stheno-Inverted-L2-13B-GPTQ
```


### Hosting It
Use host_gptq_exllama for the best speeds
```python
model_directory = sys.argv[1]
local_port = int(sys.argv[2])
gpu_id = int(sys.argv[3])
gpu_type = sys.argv[4]

```



```bash
python3 main.py
python3 host_gptq.py LosslessMegaCoder-Llama2-13B-Mini-GPTQ 7776 0
python3 host_gptq.py Mythical-Destroyer-V2-L2-13B-GPTQ 7777 1
python3 host_gptq.py Stheno-Inverted-L2-13B-GPTQ 7778 2
python3 host_gptq.py Speechless-Llama2-13B-GPTQ 7779 3

```

