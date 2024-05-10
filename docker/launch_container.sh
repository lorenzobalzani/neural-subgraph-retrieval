#!/bin/bash

image_name="neural-subgraph-retrieval"
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if the image is present
if ! docker images | grep -q "$image_name"; then
    echo "Image $image_name not found. Creating..."
    # Create the image
    docker build -t $image_name .
fi

# Run the container
docker run --rm --shm-size=32gb -v "$script_dir"/../:/workspace -v ~/.ssh:/root/.ssh --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' -it $image_name /bin/bash
