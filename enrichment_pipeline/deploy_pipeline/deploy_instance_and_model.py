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
VAST_API_KEY = "dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad"
# TODO: Handle when you are outbid
# TODO: Find the number of GPUs, and launch that many models


#Get rows of results
cmd_string = "set api-key dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad"
completed_process = subprocess.run(['./vast.py']+cmd_string.split(" "))
search_for_instances = 'search offers " num_gpus>1 reliability > 0.99 gpu_name=RTX_3090 inet_down > 200" -o "dph_total"'
search_output = subprocess.run(['./vast.py']+shlex.split(search_for_instances),stdout=subprocess.PIPE,text=True)
lines = search_output.stdout.strip().split("\n")
headers = lines[0].replace("NV Driver","NV_Driver").split()
rows = [line.split() for line in lines[1:]]

#Pick based on these criteria:
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

# Pick the first model and launch it with the right image
model_id_chosen:str = viable_models.iloc[0].loc["ID"]
disk_space_required:int = viable_models.iloc[0].loc["N"] * 20
cuda_vers = viable_models.iloc[0].loc["CUDA"] #TODO: they only go up to CUDA 11.7
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
        if target_row.loc[0,"Status"] == "running":
            break
        time.sleep(1)

#SSH into instance
get_port_and_ip_command = f'show instance {instance_id}'
get_port_and_ip_output = subprocess.run(['./vast.py']+shlex.split(check_instances_command)+['--raw'],stdout=subprocess.PIPE,text=True)
res_json = json.loads(get_port_and_ip_output.stdout)
instance_addr:str = res_json[0]['ssh_host']
instance_port:int = int(res_json[0]['ssh_port'])
print(f'Connect via SSH:\nssh -o "IdentitiesOnly=yes" -i /home/bird/dataset_enrichment/credentials/autovastai -p {instance_port} root@{instance_addr} -L 8080:localhost:8080 \n')
time.sleep(10) # Sleep for better SSH reliability

# Set up the SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys
pkey = paramiko.RSAKey.from_private_key_file("../../credentials/autovastai")

# Connect to the remote server
client.connect(
    hostname=instance_addr,
    port=instance_port,  # default port for SSH
    username='root',
    pkey=pkey,
    look_for_keys=False)
time.sleep(0.2)

# SCP and Register Private Key for Machine Account
scp = SCPClient(client.get_transport())
scp.put(files=['/home/bird/dataset_enrichment/credentials/autovastai','/home/bird/dataset_enrichment/credentials/autovastai.pub'],remote_path='/root/.ssh/')
shell = client.invoke_shell()
shell.send('eval "$(ssh-agent -s)" && ssh-add ~/.ssh/autovastai' + "\n")
client.close()
time.sleep(0.3)
client.connect(
    hostname=instance_addr,
    port=instance_port,  # default port for SSH
    username='root',
    pkey=pkey,
    look_for_keys=False)
shell = client.invoke_shell()
shell.send('ssh-keyscan github.com >> ~/.ssh/known_hosts' + "\n")
client.close()

# Connect and Install Dependancies
client.connect(
    hostname=instance_addr,
    port=instance_port,  # default port for SSH
    username='root',
    pkey=pkey,
    look_for_keys=False)
shell = client.invoke_shell()
commands = ['git clone git@github.com:surcyf123/dataset_enrichment.git','cd dataset_enrichment','pip3 install flask tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum',"sudo apt install screen","curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash","sudo apt-get install git-lfs","git lfs install","cat ./credentials/ckpt1"]
commandstr = " && ".join(commands)
shell.send(commandstr+"\n")
while not shell.recv_ready():
    time.sleep(1)
with Loader(desc="Installing Dependancies",end=f"Dependancies Ready!"):
    while "8f4d7cb3-a7a3-4e7d-9bb5-82b593196b95" not in get_tmux_content(client):
        time.sleep(1)

# Download Model
commands = ["git lfs clone https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ","cat ./credentials/ckpt2"]
commandstr = " && ".join(commands)
shell.send(commandstr+"\n")
while not shell.recv_ready():
    time.sleep(1)
with Loader(desc="Downloading Model(s)",end=f"Model(s) Ready!"):
    while "30ac9dfe-aef1-4766-a75e-0e14dd7ac27f" not in get_tmux_content(client):
        time.sleep(1)

# %%
models_uuids= []
# TODO: Wrap this in a for loop to start experiments and collect results for multiple GPUs (maybe use threading)


# Host Model
model_uuid = uuid.uuid4()
models_uuids.append(model_uuid)
start_screen_command = f"screen -S {str(model_uuid)}"
shell.send(start_screen_command + "\n")
while not shell.recv_ready():
    time.sleep(1)
time.sleep(0.3)

launch_args = {
    'model_path' : '../Llama-2-7b-Chat-GPTQ',
    'local_port' : '7777'
}

commands = ["cd enrichment_pipeline",f"python3 host_gptq_model.py {launch_args['model_path']} {launch_args['local_port']}","cat /root/dataset_enrichment/credentials/ckpt3"]
commandstr = " && ".join(commands)
shell.send(commandstr+"\n")
while not shell.recv_ready():
    time.sleep(1)
with Loader(desc=f"Launching Model: {model_uuid}",end=f"Model {model_uuid} Ready on Port {launch_args['local_port']}!"):
    while "1192ebb3-61ce-4b12-b808-1de74424432f" not in get_tmux_content(client):
        time.sleep(1)



# Print Model endpoint details

# Launch Experiment




# Upload Results to Git



# %%

