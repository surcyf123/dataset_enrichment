# %%
import subprocess
from utils import check_existence_of_filename
import pandas as pd
import shlex
import re
import time
import json
import paramiko
from scp import SCPClient
import uuid
import threading
from typing import Dict
import sys
active_branch = "journey-to-0.4"
VAST_API_KEY = "dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad"
pkey = paramiko.RSAKey.from_private_key_file("../../credentials/autovastai")



if len(sys.argv) == 2:
    use_fmt_file = bool(sys.argv[1])
    with open("fmtEXAMPLE.json", "r") as f:
        prompts = json.load(f)
        
    models_to_test = []
    prompt_formats = []
    for model in prompts:
        k,v = list(model.items())[0]
        models_to_test.append(k)
        prompt_formats.append(v)
    print("Using Default fmtEXAMPLE.json prompts")

elif len(sys.argv) == 3:
    use_fmt_file = bool(sys.argv[1])
    fmt_file_path = sys.argv[2]
    with open(fmt_file_path, "r") as f:
        prompts = json.load(f)
        
    models_to_test = []
    prompt_formats = []
    for model in prompts:
        k,v = list(model.items())[0]
        models_to_test.append(k)
        prompt_formats.append(v)
    print(f"Using prompts found at {fmt_file_path}")

elif len(sys.argv) == 1:
    use_fmt_file = False
    models_to_test=["TheBloke/MythoLogic-13B-GPTQ",
"TheBloke/MythoBoros-13B-GPTQ",
"TheBloke/Karen_theEditor_13B-GPTQ",
"TheBloke/WizardLM-13B-V1.0-Uncensored-GPTQ",
"TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ",
"TheBloke/Dolphin-Llama-13B-GPTQ",
"TheBloke/based-13b-GPTQ",
"TheBloke/CAMEL-13B-Role-Playing-Data-GPTQ",]
    print("Using hardcoded models with auto prompt discovery")


reward_endpoints = [
    # numbers after GPU type is ranked by speed of that type
    "http://47.189.79.46:50159", # 3090s1
    "http://47.189.79.46:50108",
    "http://47.189.79.46:50193",
    "http://47.189.79.46:50060"] 

assert(len(models_to_test) <= 8)
# assert(len(reward_endpoints) >= len(models_to_test))
num_gpus = len(models_to_test)

# TODO: Handle when you are outbid
# TODO: Find the number of GPUs, and launch that many models
# TODO: Wrap this in a for loop to start experiments and collect results for multiple GPUs (maybe use threading)
# vast 4


print(f"Testing Models: {', '.join(models_to_test)}")
models_no_rep_name = []
for model_name in models_to_test:
    models_no_rep_name.append(model_name.split("/")[1])

# First we launch the instance and install dependancies

# Finds all available instances
gpu_name = "RTX_4090"
cmd_string = "set api-key dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad"
completed_process = subprocess.run(['./vast.py']+cmd_string.split(" "))
search_for_instances = f'search offers " num_gpus={num_gpus} reliability > 0.70 gpu_name={gpu_name} inet_down > 200" -o "inet_down-"'
search_output = subprocess.run(['./vast.py']+shlex.split(search_for_instances),stdout=subprocess.PIPE,text=True)
lines = search_output.stdout.strip().split("\n")
headers = lines[0].replace("NV Driver","NV_Driver").split()
rows = [line.split() for line in lines[1:]]

#Pick based on criteria:
# disk space must be 20 gb per card (13b model is <10gb)
# cpu_cores_effective > gpu count
# RAM: 8gb per card
# reliability > 0.99
# num_gpus > 4
# gpu = RTX_3090
# download speed > 200
df_instances = pd.DataFrame(rows,columns=headers)
df_instances['N'] = df_instances['N'].str.replace('x','').astype(int)
df_instances['RAM'] = df_instances['RAM'].astype(float)
df_instances['vCPUs'] = df_instances['vCPUs'].astype(float)
df_instances['Disk'] = df_instances['Disk'].astype(float)
viable_models = df_instances[(df_instances['RAM'] / df_instances['N'] >= 4) & (df_instances['vCPUs'] / df_instances['N'] >= 1) * (df_instances['Disk'] > df_instances['N'] * 20)]
print(viable_models.head())


picked_row = 0 # i do this because sometimes the instances just dont work right
# Pick the first instance and launch it with the right image
model_id_chosen:str = viable_models.iloc[picked_row].loc["ID"]
number_of_gpus_in_instance = viable_models.iloc[picked_row].loc["N"]
disk_space_required:int = viable_models.iloc[0].loc["N"] * 50
cuda_vers = viable_models.iloc[0].loc["CUDA"]
launch_command = f'create instance {model_id_chosen} --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel --disk {str(disk_space_required)}'
launch_subprocess_output = subprocess.run(['./vast.py']+shlex.split(launch_command),stdout=subprocess.PIPE,text=True)
assert('success' in launch_subprocess_output.stdout) # it will fail here if there is an issue

# Find instance ID and wait for it to be done
instance_id:str =re.findall("('new_contract': )(.*)(})",launch_subprocess_output.stdout)[0][1]
print("Instance is starting...")
time.sleep(5)
while True:
    check_instances_command = f'show instances'
    
    check_instances_output = subprocess.run(['./vast.py']+shlex.split(check_instances_command),stdout=subprocess.PIPE,text=True)

    lines = check_instances_output.stdout.strip().split("\n")
    headers = lines[0].replace("Util. %","Util_%").replace("SSH Addr","SSH_Addr").replace("SSH Port","SSH_Port").replace("Net up","Net_up").replace("Net down","Net_down").split()
    rows = [line.split() for line in lines[1:]]
    df_statuses = pd.DataFrame(rows,columns=headers)
    target_row = df_statuses.loc[df_statuses['ID'] == instance_id] # Select the target row
    if target_row.iloc[0]['Status'] == "running":
        break
    time.sleep(1)
print(f"Instance {instance_id} is ready!")
#SSH into instance
get_port_and_ip_command = f'show instance {instance_id}' #TODO: It works but not really intended for it to be working this way
get_port_and_ip_output = subprocess.run(['./vast.py']+shlex.split(get_port_and_ip_command)+['--raw'],stdout=subprocess.PIPE,text=True)
res_json = json.loads(get_port_and_ip_output.stdout)
instance_addr:str = res_json['ssh_host']
instance_port:int = int(res_json['ssh_port'])
print(f'Connect via SSH:\nssh -o "IdentitiesOnly=yes" -i /home/bird/dataset_enrichment/credentials/autovastai -p {instance_port} root@{instance_addr} -L 8080:localhost:8080 -oStrictHostKeyChecking=no\n')
time.sleep(10) # Sleep for better SSH reliability

# Set up the SSH client
base_client = paramiko.SSHClient()
base_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys


while True:
    try:
        # Connect to the remote server
        base_client.connect(
            hostname=instance_addr,
            port=instance_port, 
            username='root',
            pkey=pkey,
            look_for_keys=False)
        time.sleep(0.2)
        break

    except:
        print("Retrying SSH Connection...")
        time.sleep(2)
        

# touch ~/.no_auto_tmux and then reconnect
base_client.connect(
    hostname=instance_addr,
    port=instance_port, 
    username='root',
    pkey=pkey,
    look_for_keys=False)
base_shell = base_client.invoke_shell(width=120, height=30)
base_shell.send('touch ~/.no_auto_tmux'+"\n")
time.sleep(1)
while not base_shell.recv_ready():
    time.sleep(1)
base_client.close()



check_base_ckpt_client = paramiko.SSHClient()
check_base_ckpt_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys

check_base_ckpt_client.connect(
    hostname=instance_addr,
    port=instance_port, 
    username='root',
    pkey=pkey,
    look_for_keys=False)

# SCP and Register Private Key for Machine Account on a new client with tmux

install_dep_client = paramiko.SSHClient()
install_dep_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys

install_dep_client.connect(
    hostname=instance_addr,
    port=instance_port, 
    username='root',
    pkey=pkey,
    look_for_keys=False)
dep_shell = install_dep_client.invoke_shell(width=120, height=30)
dep_shell.send('tmux new -s install_deps' + "\n")

scp = SCPClient(install_dep_client.get_transport())
scp.put(files=['/home/bird/dataset_enrichment/credentials/autovastai','/home/bird/dataset_enrichment/credentials/autovastai.pub'],remote_path='/root/.ssh/')
dep_shell.send('chmod 600 ~/.ssh/autovastai && chmod 600 ~/.ssh/autovastai.pub' + "\n")
time.sleep(0.1)
dep_shell.send('eval "$(ssh-agent -s)" && ssh-add ~/.ssh/autovastai' + "\n")

time.sleep(0.1)
dep_shell.send('ssh-keyscan github.com >> ~/.ssh/known_hosts' + "\n")
# Setup folder for dataset upload
time.sleep(0.1)
dep_shell.send("cd /root/" + "\n")
time.sleep(0.1)
dep_shell.send("git clone git@github.com:surcyf123/quantized_reward_results.git"+"\n")
while not dep_shell.recv_ready():
    time.sleep(1)
dep_shell.send("git config --global user.name 'AutoVastAI' && git config --global user.email 'deckenball@gmail.com'"+"\n")


# Connect and Install Dependancies
time.sleep(0.3)
commands = ['git clone git@github.com:surcyf123/dataset_enrichment.git','cd /root/dataset_enrichment/','pip3 install --upgrade Pillow',f'git checkout {active_branch}','pip3 install flask tqdm torch tiktoken transformers peft accelerate torchvision torchaudio auto-gptq optimum boto3 uvicorn vllm pydantic fastapi',"sudo apt install screen","curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash","sudo apt-get install git-lfs","git lfs install","pip3 install flash-attn --no-build-isolation", "mkdir /root/ckpts","touch /root/ckpts/ckpt1"]

# commands = ['git clone git@github.com:surcyf123/dataset_enrichment.git','cd /root/dataset_enrichment/','pip3 install --upgrade Pillow',f'git checkout {active_branch}','pip3 install flask tqdm torch tiktoken transformers peft accelerate torchvision torchaudio auto-gptq optimum',"sudo apt install screen","curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash","sudo apt-get install git-lfs","git lfs install","pip3 install flash-attn --no-build-isolation", "git clone https://github.com/chu-tianxiang/vllm-gptq.git", "cd vllm-gptq", "pip3 install -e .","cat /root/dataset_enrichment/credentials/ckpt1"]
commandstr = " && ".join(commands)
dep_shell.send(commandstr+"\n")
while not dep_shell.recv_ready():
    time.sleep(1)
print("Installing Dependancies")
while "thefileishereandthisisnotafluke" not in check_existence_of_filename("ckpt1",check_base_ckpt_client):
    time.sleep(1)
print("Dependancies Ready!")
# %% 
commands = ["pip3 install --upgrade nvitop", "nvitop"]
commandstr = " && ".join(commands)
dep_shell.send(commandstr+"\n")


# Init UUIDs
# ------------------------------------------- I should just initialize the clients and shells, then pass it into the threading --------------------
# Start a tmux session

experiment_clients: Dict[int,paramiko.client.SSHClient] = {} # dict to store all the experiments active sessions' clients
experiment_shells: Dict[int,paramiko.Channel] = {} # dict to store all the experiments active sessions' shells


model_clients: Dict[int,paramiko.client.SSHClient] = {} # dict to store all the models active sessions' clients
model_shells: Dict[int,paramiko.Channel] = {} # dict to store all the models active sessions' shells

checker_clients = {}

base_port = 7777
for i in range(len(models_to_test)):
    chosen_experiment_model_name = models_to_test[i]
    chosen_experiment_model_port = base_port + i
    # pass in the clients and shells, and use the experiment id to index them.

    # Initialize the shell and tmux session to be used
    model_clients[i] = paramiko.SSHClient()
    model_clients[i].set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys
    experiment_clients[i] = paramiko.SSHClient()
    experiment_clients[i].set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys

    model_clients[i].connect(
        hostname=instance_addr,
        port=instance_port, 
        username='root',
        pkey=pkey,
        look_for_keys=False)
    
    experiment_clients[i].connect(
        hostname=instance_addr,
        port=instance_port, 
        username='root',
        pkey=pkey,
        look_for_keys=False)
    
    model_shells[i] = model_clients[i].invoke_shell(width=120, height=30)
    model_shells[i].send(f'tmux new -s model_{str(i)}' + "\n")
    
    experiment_shells[i] = experiment_clients[i].invoke_shell(width=120, height=30)
    experiment_shells[i].send(f'tmux new -s experiment_{str(i)}' + "\n")
    
    checker_clients[i] = paramiko.SSHClient()
    checker_clients[i].set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys

    checker_clients[i].connect(
        hostname=instance_addr,
        port=instance_port, 
        username='root',
        pkey=pkey,
        look_for_keys=False)
    
    
print(f"Shells and Clients and Checkers Initialized: {', '.join(str(a) for a in list(experiment_clients.keys()))}")

# ----------------------------------------------- This can be outside the threading ^ ---------------------------------------------

def download_model_run_experiment_upload_results(chosen_experiment_model_name,chosen_experiment_model_port,experiment_id,model_clients,model_shells,experiment_clients,experiment_shells,checker_clients):
    # Download Model
    commands = [f"mkdir -p /root/results/{experiment_id}",f"mkdir -p /root/results/{experiment_id}/performance_summaries",f"mkdir -p /root/results/{experiment_id}/raw_results","cd /root/dataset_enrichment/enrichment_pipeline",f"git lfs clone https://huggingface.co/TheBloke/{chosen_experiment_model_name}",f"touch /root/ckpts/{experiment_id}_ckpt2"]
    # commands = ['cat /root/dataset_enrichment/credentials/ckpt2']
    commandstr = " && ".join(commands)
    model_shells[experiment_id].send(commandstr+"\n")
    while not model_shells[experiment_id].recv_ready():
        time.sleep(1)
    print(f"{experiment_id}: Downloading Model(s)")
    
    while "thefileishereandthisisnotafluke" not in check_existence_of_filename(f"{experiment_id}_ckpt2",checker_clients[experiment_id]):
        time.sleep(1)
    
    print(f"Model {experiment_id} Downloaded!")
    
    # Start Screen

    launch_args = {
        'model_path' : chosen_experiment_model_name,
        'local_port' : chosen_experiment_model_port,
        'gpuID' : experiment_id,
        'reward_endpoint' : reward_endpoints[experiment_id//2 % len(reward_endpoints)]
    }
    # Run Model
    commands = ["cd /root/dataset_enrichment/enrichment_pipeline",f"python3 host_awq_model_vllm.py --model {launch_args['model_path']} --port {launch_args['local_port']} --gpu_id {launch_args['gpuID']} --gpu_type={gpu_name.replace('RTX_','')}"]
    commandstr = " && ".join(commands)
    model_shells[experiment_id].send(commandstr+"\n")
    while not model_shells[experiment_id].recv_ready():
        time.sleep(1)
    print(f"{experiment_id}:Launching Model: {experiment_id}")
    while "thefileishereandthisisnotafluke" not in check_existence_of_filename(f"{experiment_id}_ckpt3",checker_clients[experiment_id]):
        time.sleep(1)
        
    print(f"Model {experiment_id} Ready on Port {launch_args['local_port']}!")

    # 
    # Launch Experiment
    # Start Screen
    time.sleep(5)

    # Run Experiment
    experiment_shells[experiment_id].send("cd /root/dataset_enrichment/enrichment_pipeline"+"\n")
    time.sleep(0.1)
    experiment_shells[experiment_id].send('eval "$(ssh-agent -s)" && ssh-add ~/.ssh/autovastai'+"\n")
    time.sleep(0.1)
    experiment_shells[experiment_id].send(f"python3 conduct_experiment_on_model.py {launch_args['model_path']} {launch_args['local_port']} {experiment_id} {launch_args['reward_endpoint']} {gpu_name} '{prompt_formats[experiment_id] if use_fmt_file else ''}'"+"\n")
    time.sleep(0.1)
    
    print(f"{experiment_id}:Running Experiment: {experiment_id}")
    while "thefileishereandthisisnotafluke" not in check_existence_of_filename(f"{experiment_id}_ckpt4",checker_clients[experiment_id]):
        time.sleep(1)
    print(f"Experiment {experiment_id} Done! Pushing Results...")

    # Get the experiment averages
    
    


    
    
    # print(f"All Results Pushed for Experiment: {experiment_id}")

    # Close both screens for the model, and for the experiment running
    

# (chosen_experiment_model_name,chosen_experiment_model_port,experiment_id,clients,shells):
# Get all the different threads going
threads = []
for i in range(len(models_to_test)):
    chosen_experiment_model_name = models_no_rep_name[i]
    chosen_experiment_model_port = base_port + i
    
    t = threading.Thread(target=download_model_run_experiment_upload_results, args=(chosen_experiment_model_name, chosen_experiment_model_port, i,model_clients,model_shells,experiment_clients,experiment_shells,checker_clients))

    t.start()
    threads.append(t)
time.sleep(0.3)
print("All threads are running")   



# Start the next one somewhere somehow

# %%
