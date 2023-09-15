#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <gpu_type> <model_choices>"
    echo "gpu_type should be either '3090' or '4090'."
    echo "model_choices should be one of 'models8x1', 'models8x2', 'models8x3'."
    exit 1
fi

gpu_type=$1
model_choice=$2

# Validate GPU type
if [[ "$gpu_type" != "3090" && "$gpu_type" != "4090" ]]; then
    echo "Invalid gpu_type. Choose either '3090' or '4090'."
    exit 1
fi

# Define model arrays using indexed arrays
models8x1=("Speechless-Llama2-13B-GPTQ" "Mythical-Destroyer-V2-L2-13B-GPTQ" "Luban-13B-GPTQ" "Stheno-Inverted-L2-13B-GPTQ" "LosslessMegaCoder-Llama2-13B-Mini-GPTQ" "OpenOrca-Platypus2-13B-GPTQ" "Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GPTQ" "Llama2-13B-MegaCode2-OASST-GPTQ")
models8x2=("Huginn-13B-v4-GPTQ" "UndiMix-v2-13B-GPTQ" "Huginn-13B-v4.5-GPTQ" "Huginn-v3-13B-GPTQ" "LoKuS-13B-GPTQ" "PuddleJumper-13B-GPTQ" "Speechless-Llama2-13B-GPTQ" "Mythical-Destroyer-V2-L2-13B-GPTQ")
models8x3=("Stheno-Inverted-L2-13B-GPTQ" "LosslessMegaCoder-Llama2-13B-Mini-GPTQ" "OpenOrca-Platypus2-13B-GPTQ" "Speechless-Llama2-13B-GPTQ" "Llama2-13B-MegaCode2-OASST-GPTQ" "Huginn-13B-v4-GPTQ" "UndiMix-v2-13B-GPTQ" "Huginn-13B-v4.5-GPTQ")

# Use a case statement to assign models based on model_choice
case "$model_choice" in
    "models8x1")
        models=("${models8x1[@]}")
        ;;
    "models8x2")
        models=("${models8x2[@]}")
        ;;
    "models8x3")
        models=("${models8x3[@]}")
        ;;
    *)
        echo "Invalid model_choices. Choose one of 'models8x1', 'models8x2', or 'models8x3'."
        exit 1
        ;;
esac

# Change to the home directory
cd ~/

# Clone each model repository if not already present
for model in "${models[@]}"; do
    if [ ! -d "$model" ]; then
        git clone "https://huggingface.co/TheBloke/$model"
    else
        echo "$model already exists. Skipping clone."
    fi
done

cd ~/dataset_enrichment/enrichment_pipeline/production_deployment
# Define port range
start_port=30000
end_port=$((start_port + ${#models[@]} - 1))
ports=($(seq $start_port $end_port))

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
    model="~/${models[$i]}"  # The model path is relative to the parent directory
    port=${ports[$i]}
    gpu_id=$i  # This assumes you start with GPU ID 0 and increase by 1 for each model

    # Append the URL to the urls array
    urls+=("http://$ip:$port")

    echo "Starting model $model on port $port with GPU ID $gpu_id..."
    pm2 start ~/dataset_enrichment/enrichment_pipeline/prod/host_gptq.py --name "${models[$i]}" --interpreter python3 -- "$model" "$port" "$gpu_id" "$gpu_type"
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