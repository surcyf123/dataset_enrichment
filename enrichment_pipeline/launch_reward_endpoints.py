#!/bin/bash

START_PORT=30000
NUM_INSTANCES=4 # cChange to amount of GPUs you have / num reward_instances you want to run.
NAME=$START_PORT

# Start NUM_INSTANCES instances
for ((i=0; i<$NUM_INSTANCES; i++))
do
    # Calculate the GPU ID and port for this instance
    GPU_ID=$i
    PORT=$((START_PORT + i))

    # Start the process with pm2
    pm2 start --name "${PORT}" --time --interpreter=python3 /root/dataset_enrichment/enrichment_pipeline/reward_endpoint.py -- --gpu $GPU_ID --port $PORT
done

