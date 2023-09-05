def get_tmux_content(ssh_client):
    command = "tmux capture-pane -p"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    return stdout.read().decode('utf-8')