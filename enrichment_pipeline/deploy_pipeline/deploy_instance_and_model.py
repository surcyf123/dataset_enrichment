# %%
import subprocess
from loader import Loader
from utils import get_tmux_content
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
pkey = paramiko.RSAKey.from_private_key_file("../../credentials/autovastai")
VAST_API_KEY = "dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad"
# TODO: Handle when you are outbid
# TODO: Find the number of GPUs, and launch that many models
# TODO: Wrap this in a for loop to start experiments and collect results for multiple GPUs (maybe use threading)

# Define which models we want to test



# First we launch the instance and install dependancies

# Finds all available instances
cmd_string = "set api-key dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad"
completed_process = subprocess.run(['./vast.py']+cmd_string.split(" "))
search_for_instances = 'search offers " num_gpus>1 reliability > 0.99 gpu_name=RTX_3090 inet_down > 200" -o "dph_total"'
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

# Pick the first instance and launch it with the right image
model_id_chosen:str = viable_models.iloc[3].loc["ID"]
number_of_gpus_in_instance = viable_models.iloc[3].loc["N"]
disk_space_required:int = viable_models.iloc[0].loc["N"] * 50
cuda_vers = viable_models.iloc[0].loc["CUDA"]
launch_command = f'create instance {model_id_chosen} --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel --disk {str(disk_space_required)}'
launch_subprocess_output = subprocess.run(['./vast.py']+shlex.split(launch_command),stdout=subprocess.PIPE,text=True)
assert('success' in launch_subprocess_output.stdout) # it will fail here if there is an issue

# Find instance ID and wait for it to be done
instance_id:str =re.findall("('new_contract': )(.*)(})",launch_subprocess_output.stdout)[0][1]
with Loader(desc="Instance is starting...",end=f"Instance {instance_id} is ready!"):
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

#SSH into instance
get_port_and_ip_command = f'show instance {instance_id}' #TODO: It works but not really intended for it to be working this way
get_port_and_ip_output = subprocess.run(['./vast.py']+shlex.split(get_port_and_ip_command)+['--raw'],stdout=subprocess.PIPE,text=True)
res_json = json.loads(get_port_and_ip_output.stdout)
instance_addr:str = res_json['ssh_host']
instance_port:int = int(res_json['ssh_port'])
print(f'Connect via SSH:\nssh -o "IdentitiesOnly=yes" -i /home/bird/dataset_enrichment/credentials/autovastai -p {instance_port} root@{instance_addr} -L 8080:localhost:8080 \n')
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
base_shell = base_client.invoke_shell()
base_shell.send('touch ~/.no_auto_tmux'+"\n")
while not base_shell.recv_ready():
    time.sleep(1)
base_client.close()


# SCP and Register Private Key for Machine Account on a new client with tmux

install_dep_client = paramiko.SSHClient()
install_dep_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys

install_dep_client.connect(
    hostname=instance_addr,
    port=instance_port, 
    username='root',
    pkey=pkey,
    look_for_keys=False)
dep_shell = install_dep_client.invoke_shell()
dep_shell.send('tmux new -s install_deps' + "\n")

scp = SCPClient(install_dep_client.get_transport())
scp.put(files=['/home/bird/dataset_enrichment/credentials/autovastai','/home/bird/dataset_enrichment/credentials/autovastai.pub'],remote_path='/root/.ssh/')
dep_shell.send('chmod 600 ~/.ssh/autovastai && chmod 600 ~/.ssh/autovastai.pub' + "\n")
time.sleep(0.3)
dep_shell.send('eval "$(ssh-agent -s)" && ssh-add ~/.ssh/autovastai' + "\n")

time.sleep(0.3)
dep_shell.send('ssh-keyscan github.com >> ~/.ssh/known_hosts' + "\n")
# Connect and Install Dependancies
time.sleep(0.3)
commands = ['git clone git@github.com:surcyf123/dataset_enrichment.git','cd /root/dataset_enrichment/','pip3 install flask tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum',"sudo apt install screen","curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash","sudo apt-get install git-lfs","git lfs install","cat root/dataset_enrichment/credentials/ckpt1"]
commandstr = " && ".join(commands)
dep_shell.send(commandstr+"\n")
while not dep_shell.recv_ready():
    time.sleep(1)
with Loader(desc="Installing Dependancies",end=f"Dependancies Ready!"):
    while "8f4d7cb3-a7a3-4e7d-9bb5-82b593196b95" not in get_tmux_content(install_dep_client):
        time.sleep(1)

dep_shell.send('\x02')
time.sleep(0.1)
dep_shell.send('x\n')
time.sleep(0.1)
dep_shell.send('y\n')
install_dep_client.close()

# Init UUIDs
# ------------------------------------------- I should just initialize the clients and shells, then pass it into the threading --------------------
# Start a tmux session
clients: Dict[int,paramiko.client.SSHClient] = {} # dict to store all the active sessions' clients
shells: Dict[int,paramiko.Channel] = {} # dict to store all the active sessions' shells
# TheBloke/Pygmalion-2-13B-GPTQ    #7777 (int)          0
models_to_test = ['TheBloke/Pygmalion-2-13B-GPTQ','TheBloke/13B-Thorns-L2-GPTQ','TheBloke/Kimiko-13B-GPTQ','TheBloke/OpenBuddy-Llama2-13B-v11.1-GPTQ']
base_port = 7777
for i in range(number_of_gpus_in_instance):
    chosen_experiment_model_name = models_to_test[i]
    chosen_experiment_model_port = base_port + i
    # pass in the clients and shells, and use the experiment id to index them.

    # Initialize the shell and tmux session to be used
    clients[i] = paramiko.SSHClient()
    clients[i].set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys

    clients[i].connect(
        hostname=instance_addr,
        port=instance_port, 
        username='root',
        pkey=pkey,
        look_for_keys=False)
    shells[i] = clients[i].invoke_shell()
    shells[i].send(f'tmux new -s experiment_{str(i)}' + "\n")
print(f"Shells and Clients Initialized: {', '.join(str(a) for a in list(clients.keys()))}")

# ----------------------------------------------- This can be outside the threading ^ ---------------------------------------------
# %%
def download_model_run_experiment_upload_results(chosen_experiment_model_name,chosen_experiment_model_port,experiment_id,clients,shells):
    # Download Model
    commands = ["cd /root/dataset_enrichment/",f"git lfs clone https://huggingface.co/{chosen_experiment_model_name}","cat /root/dataset_enrichment/credentials/ckpt2"]
    commandstr = " && ".join(commands)
    shells[experiment_id].send(commandstr+"\n")
    while not shells[experiment_id].recv_ready():
        time.sleep(1)
    with Loader(desc=f"{experiment_id}: Downloading Model(s)",end=f"Model(s) Ready!"):
        while "30ac9dfe-aef1-4766-a75e-0e14dd7ac27f" not in get_tmux_content(clients[experiment_id]):
            time.sleep(1)

    # I need to launch this in its own screen
    base_uuid = str(uuid.uuid4())
    model_uuid = f"{experiment_id}:MOD:"+base_uuid
    experiment_uuid = f"{experiment_id}:EXP:"+base_uuid

    # Start Screen
    start_screen_command = f"screen -S {model_uuid}"
    shells[experiment_id].send(start_screen_command + "\n")
    while not shells[experiment_id].recv_ready():
        time.sleep(1)
    time.sleep(0.3)

    launch_args = {
        'model_path' : chosen_experiment_model_name,
        'local_port' : chosen_experiment_model_port,
        'gpuID' : experiment_id
    }
    # Run Model
    commands = ["cd /root/dataset_enrichment/enrichment_pipeline",f"python3 host_gptq_model.py {launch_args['model_path']} {launch_args['local_port']} {launch_args['gpuID']}"]
    commandstr = " && ".join(commands)
    shells[experiment_id].send(commandstr+"\n")
    while not shells[experiment_id].recv_ready():
        time.sleep(1)
    with Loader(desc=f"{experiment_id}:Launching Model: {model_uuid}",end=f"Model {model_uuid} Ready on Port {launch_args['local_port']}!"):
        while "Serving Flask app" not in get_tmux_content(clients[experiment_id]):
            time.sleep(1)

    # Detatch Screen
    shells[experiment_id].send('\x01')
    time.sleep(0.1)
    shells[experiment_id].send('d\n')
    time.sleep(1)
    shells[experiment_id].send('cd /root/dataset_enrichment/'+"\n")

    # 
    # Launch Experiment
    # Start Screen
    time.sleep(1)
    start_screen_command = f"screen -S {experiment_uuid}"
    shells[experiment_id].send(start_screen_command + "\n")
    while not shells[experiment_id].recv_ready():
        time.sleep(1)
    time.sleep(0.3)

    # Run Experiment
    commands = ["cd /root/dataset_enrichment/enrichment_pipeline",f"python3 conduct_experiment_on_model.py {launch_args['model_path']} {launch_args['local_port']} {experiment_uuid}"]
    commandstr = " && ".join(commands)
    shells[experiment_id].send(commandstr+"\n")
    while not shells[experiment_id].recv_ready():
        time.sleep(1)
    with Loader(desc=f"{experiment_id}:Running Experiment: {model_uuid}",end=f"Model {model_uuid} Ready on Port {launch_args['local_port']}!"):
        while "Experiment Complete" not in get_tmux_content(clients[experiment_id]):
            time.sleep(1)

    
    # Get the experiment averages



    # Upload Results to Git

    # Close both screens for the model, and for the experiment running
        
    # Detatch Screen
    shells[experiment_id].send('\x01')
    time.sleep(0.1)
    shells[experiment_id].send('d\n')
    while not shells[experiment_id].recv_ready():
        time.sleep(1)
    
    

# (chosen_experiment_model_name,chosen_experiment_model_port,experiment_id,clients,shells):
# Get all the different threads going
threads = []
for i in range(number_of_gpus_in_instance):
    chosen_experiment_model_name = models_to_test[i]
    chosen_experiment_model_port = base_port + i
    
    t = threading.Thread(target=download_model_run_experiment_upload_results, args=(chosen_experiment_model_name, chosen_experiment_model_port, i,clients,shells))

    t.start()
    threads.append(t)
time.sleep(0.3)
print("All threads are running")   



# Start the next one somewhere somehow

# %%
