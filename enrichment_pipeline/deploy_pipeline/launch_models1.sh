#!/bin/bash

# List of model strings
models=("Luban-13B-GPTQ" "Mythical-Destroyer-V2-L2-13B-GPTQ" "Speechless-Llama2-13B-GPTQ" "Stheno-Inverted-L2-13B-GPTQ")

# List of ports
ports=(30000 30001 30002 30003) 

# Ensure that the number of models and ports are the same
if [[ ${#models[@]} -ne ${#ports[@]} ]]; then
    echo "Error: The number of models and ports must be the same."
    exit 1
fi

# Get public IP
ip=$(curl -s https://ifconfig.me)

# Initialize an empty array for URLs
declare -a urls

# Start PM2 processes
for i in "${!models[@]}"; do
    model="../${models[$i]}"  # The model path is relative to the parent directory
    port=${ports[$i]}
    gpu_id=$i  # This assumes you start with GPU ID 0 and increase by 1 for each model

    # Append the URL to the urls array
    urls+=("http://$ip:$port")

    echo "Starting model $model on port $port with GPU ID $gpu_id..."
    pm2 start host_gptq.py --name "${models[$i]}" --interpreter python3 -- "$model" "$port" "$gpu_id"
done

# Print the list of URLs as a Python list
echo "All models started!"
echo -n "["
for i in "${!urls[@]}"; do
    if [[ $i -ne 0 ]]; then
        echo -n ","
    fi
    echo -n "\"${urls[$i]}\""
done
echo "]"