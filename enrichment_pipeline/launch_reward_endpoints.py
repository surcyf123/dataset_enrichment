#!/bin/bash

# if pm2 isn't installed
# curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash - && \
# sudo apt-get install -y nodejs && \
# sudo npm install -g npm && \
# npm install -g pm2

# if broken packages
# sudo apt-get purge nodejs npm
# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
# source ~/.bashrc
# nvm install 14.21.3
# npm install -g pm2


START_PORT=30000
NUM_INSTANCES=4 # Change to the number of GPUs you have / number of reward_instances you want to run.
declare -i EXTERNAL_PORT=40357 #change this if you want to get a list of the endpoint urls
NAME=$START_PORT

# Get the external IP address
EXTERNAL_IP=$(curl -s ifconfig.me)
urls=() # Initialize an empty array for URLs

url_list="["

# Start NUM_INSTANCES instances
for ((i=0; i<$NUM_INSTANCES; i++))
do
    # Calculate the GPU ID and port for this instance
    GPU_ID=$i
    PORT=$((START_PORT + i))

    # Start the process with pm2
    pm2 start --name "${PORT}" --time --interpreter=python3 /root/dataset_enrichment/enrichment_pipeline/reward_endpoint.py -- --gpu $GPU_ID --port $PORT

    # Append the URL for this reward endpoint to the urls array
    url_list+="\"http://$EXTERNAL_IP:$EXTERNAL_PORT\","
    EXTERNAL_PORT+=1
done

# Remove the trailing comma and close the list
url_list="${url_list%,}]"

# Print the URLs
echo $url_list

