# Prod Endpoint

### Calling It

```
http://89.177.60.89:40885/process_question
{"prompt" : "proompt", "completions_needed" : 4, "max_time_allowed" : 30,"question_value" : 3}
```

### Installing Deps
```python
a = 'pip3 install --upgrade Pillow','pip3 install flask tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum',"sudo apt install screen","

# git lfs

'curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash","sudo apt-get install git-lfs","git lfs install"
```


```
git lfs clone https://huggingface.co/TheBloke/Stheno-Inverted-L2-13B-GPTQ
```


### Hosting It

```
python3 main.py
python3 host_gptq.py LosslessMegaCoder-Llama2-13B-Mini-GPTQ 7776 0
python3 host_gptq.py Mythical-Destroyer-V2-L2-13B-GPTQ 7777 1
python3 host_gptq.py Stheno-Inverted-L2-13B-GPTQ 7778 2
python3 host_gptq.py Speechless-Llama2-13B-GPTQ 7779 3

```

