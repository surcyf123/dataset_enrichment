# %%
import subprocess
API_KEY = "dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad"

cmd_string = "set api-key dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad"

completed_process = subprocess.run(['./vast.py']+cmd_string.split(" "))
# %%
import pandas as pd
import shlex
search_for_instances = 'search offers " num_gpus>4 reliability > 0.99 gpu_name=RTX_3090 inet_down > 200" -o "dph_total"'

search_output = subprocess.run(['./vast.py']+shlex.split(search_for_instances),stdout=subprocess.PIPE,text=True)

lines = search_output.stdout.strip().split("\n")
headers = lines[0].replace("NV Driver","NV_Driver").split()
rows = [line.split() for line in lines[1:]]



# %%
df_instances = pd.DataFrame(rows,columns=headers)
df_instances['N'] = df_instances['N'].str.replace('x','').astype(int)
df_instances['RAM'] = df_instances['RAM'].astype(float)
df_instances['vCPUs'] = df_instances['vCPUs'].astype(float)
df_instances['Disk'] = df_instances['Disk'].astype(float)

viable_models = df_instances[(df_instances['RAM'] / df_instances['N'] >= 4) & (df_instances['vCPUs'] / df_instances['N'] >= 1) * (df_instances['Disk'] > df_instances['N'] * 20)]

# ./vast.py create instance [id]
# find the number of GPUs, and launch that many models
# disk space must be 20 gb per card (13b model is <10gb)
# cpu_cores_effective > gpu count
# RAM: 8gb per card
# reliability > 0.99
# num_gpus > 4
# gpu = RTX_3090
# download speed > 200
# %%
# Pick the first model and launch it with the right image

model_id_chosen:str = viable_models.iloc[0].loc["ID"]
disk_space_required:int = viable_models.iloc[0].loc["N"] * 20
cuda_vers = viable_models.iloc[0].loc["CUDA"] #TODO: they only go up to CUDA 11.7

launch_command = f'create instance {model_id_chosen} --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel --disk {str(disk_space_required)}'

launch_subprocess_output = subprocess.run(['./vast.py']+shlex.split(launch_command),stdout=subprocess.PIPE,text=True)
assert('success' in launch_subprocess_output.stdout)
print(launch_subprocess_output)
# %%
# Get the instance ID
import re
instance_id:str =re.findall("('new_contract': )(.*)(})",launch_subprocess_output.stdout)[0][1]


# Wait for it to be done
# %%
import time
while True:
    check_instances_command = f'show instances'
    
    check_instances_output = subprocess.run(['./vast.py']+shlex.split(check_instances_command),stdout=subprocess.PIPE,text=True)

    lines = check_instances_output.stdout.strip().split("\n")
    headers = lines[0].replace("Util. %","Util_%").replace("SSH Addr","SSH_Addr").replace("SSH Port","SSH_Port").replace("Net up","Net_up").replace("Net down","Net_down").split()
    rows = [line.split() for line in lines[1:]]
    df_statuses = pd.DataFrame(rows,columns=headers)
    # print(df_statuses)

    target_row = df_statuses.loc[df_statuses['ID'] == instance_id]
    if target_row.loc[0,"Status"] == "running":
        print(f"Instance {instance_id} is ready")
        break
    print("instance is starting...")
    time.sleep(5)



# df_statuses.iloc[0].
#SSH into instance

#get connection details
# %%
import json
get_port_and_ip_command = f'show instance {instance_id}'
get_port_and_ip_output = subprocess.run(['./vast.py']+shlex.split(check_instances_command)+['--raw'],stdout=subprocess.PIPE,text=True)
print()
res_json = json.loads(get_port_and_ip_output.stdout)
instance_addr:str = res_json[0]['public_ipaddr']
instance_port:int = int(res_json[0]['ssh_port'])

# Check if direct_port_end is -1, if it is, we connect to the proxy
if instance_port == -1:
    instance_addr:str = res_json[0]['ssh_host']
    instance_port:int = int(res_json[0]['20800'])



# %%

import paramiko

# Set up the SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys

# Connect to the remote server
client.connect(
    hostname=instance_addr,
    port=instance_port,  # default port for SSH
    username='root',
    key_filename='../../credentials/autovastai'
)
# %%
# Execute command
# Install dependancies and download model
commands = ['pip3 install tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum','cd /root/', 'mkdir pineapple', 'cd pineapple',"curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash","sudo apt-get install git-lfs","git lfs install","git lfs clone https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ"]

shell = client.invoke_shell()
for cmd in commands:
    print(f"Invoking: {cmd}")
    shell.send(cmd+"\n")
    while not shell.recv_ready():  # Wait for the command to complete
        pass
    output = shell.recv(2048).decode()  # adjust the byte number if needed
    print(output)


# Close the SSH client
client.close()


 #ssh $(./vastai ssh-url 6899217)


# %%

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # This is to set the policy to use when connecting to servers without known host keys

# Connect to the remote server
client.connect(
    hostname=instance_addr,
    port=instance_port,  # default port for SSH
    username='root',
    key_filename='/home/bird/.ssh/id_rsa'
)

# Execute command
# Install dependancies and download model
commands = ["ls"]

shell = client.invoke_shell()
for cmd in commands:
    print(f"Invoking: {cmd}")
    shell.send(cmd+"\n")
    while not shell.recv_ready():  # Wait for the command to complete
        pass
    output = shell.recv(2048).decode()  # adjust the byte number if needed
    print(output)


# Close the SSH client
client.close()
# %%

# SSH in and run some launch script 
# i need to pass in a model for each gpu
# pass in the hyperparams and stuff
# pull repo
# install deps

# TODO: Handle when you are outbid

'''
pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

'''


# %%
