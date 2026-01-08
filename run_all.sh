#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output

# Array of possible sizes (one must be 500x500)
sizes=("500 500" "500 500" "500 500" "500 500" "500 500" "500 500" "500 500")

# Iterate over all txt files in scenes folder
for scene_file in scenes/*.txt; do
    # Skip if no txt files found
    [ -e "$scene_file" ] || continue
    
    # Extract scene name without path and extension
    scene_name=$(basename "$scene_file" .txt)
    
    echo "=========================================="
    echo "Processing: $scene_name"
    echo "=========================================="
    
    # Randomly decide which run gets 500x500
    # 0 = sequential gets 500x500, 1 = vectorized gets 500x500
    fixed_run=$((RANDOM % 2))
    
    # Get a random size for the other run
    random_index=$((RANDOM % ${#sizes[@]}))
    random_size="${sizes[$random_index]}"
    read -r rand_w rand_h <<< "$random_size"
    
    # Run sequential
    if [ $fixed_run -eq 0 ]; then
        seq_w=500
        seq_h=500
    else
        seq_w=$rand_w
        seq_h=$rand_h
    fi
    
    seq_output="output/${scene_name}_seq_${seq_w}_${seq_h}.png"
    echo "Running sequential: ${seq_w}x${seq_h}"
    # python ray_tracer.py "$scene_file" "$seq_output" --width "$seq_w" --height "$seq_h" --sequential
    
    # Get another random size for vectorized if sequential got 500x500
    if [ $fixed_run -eq 0 ]; then
        random_index=$((RANDOM % ${#sizes[@]}))
        random_size="${sizes[$random_index]}"
        read -r vec_w vec_h <<< "$random_size"
    else
        vec_w=500
        vec_h=500
    fi
    
    vec_output="output/${scene_name}_vectorized_${vec_w}_${vec_h}.png"
    echo "Running vectorized: ${vec_w}x${vec_h}"
    python ray_tracer.py "$scene_file" "$vec_output" --width "$vec_w" --height "$vec_h"
    
    echo ""
done

echo "=========================================="
echo "All scenes processed!"
echo "Output saved to: output/"
echo "=========================================="