from typing import Tuple
import time
def get_tmux_content(ssh_client):
    command = "tmux capture-pane -p"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    return stdout.read().decode('utf-8')


def refresh_tmux_pane(ssh_client,ssh_shell):
    ssh_shell.send('\x02r')
    time.sleep(1)

def check_existence_of_filename(filename,client):
    stdin, stdout, stderr = client.exec_command(f'[ -f "/root/ckpts/{filename}" ] && echo "thefileishereandthisisnotafluke"')
    time.sleep(1)
    return stdout.read().decode('utf-8')

def stub_call_model_with_params(prompt:str,num_tokens:int,temperature:float, top_p:float, top_k:int, repetition_penalty:float) -> Tuple[str,float]:
    '''Returns the generated text, along with how long it took to execute'''
    data = {
    "prompt": prompt,
    "max_new_tokens": num_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "repetition_penalty": repetition_penalty,
    "stopwords": []
}
    return "stub",0.05