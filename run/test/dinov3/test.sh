#!/bin/bash

export CUDA_VISIBLE_DEVICES=9
NUM_WORKERS=4

POSTPROCESSORS=("msp" "openmax" "mds" "gram" "react" "klm" "vim" "knn" "sirc_ml") 

CHECKPOINTS=(
    "./OpenOOD/Dinov3-L-ImageNet1k/CE/best.pth"
)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


for post in "${POSTPROCESSORS[@]}"; do
    for ckpt in "${CHECKPOINTS[@]}"; do
        parent_dir=$(basename "$(dirname "${ckpt}")")

        s_id="s0"  

        OUTPUT_DIR="Dinov3-L-ImageNet1k/${parent_dir}/${post}/${s_id}"
        mkdir -p "${OUTPUT_DIR}"

        echo "Running postprocessor: ${post}, checkpoint: ${ckpt}, output_dir: ${OUTPUT_DIR}"

        PYTHONPATH='.':$PYTHONPATH \
        python openood/main.py \
            --config openood/configs/datasets/imagenet/imagenet.yml \
            openood/configs/datasets/imagenet/imagenet_ood.yml \
            openood/configs/networks/dinov3_l.yml \
            openood/configs/pipelines/test/test_ood.yml \
            openood/configs/preprocessors/base_preprocessor.yml \
            openood/configs/postprocessors/${post}.yml \
            --num_workers ${NUM_WORKERS} \
            --network.checkpoint "${ckpt}" \
            --network.name dinov3_l \
            --output_dir "${OUTPUT_DIR}"
    done
done