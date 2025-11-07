#!/bin/bash
# sh scripts/ood/run_all.sh

export CUDA_VISIBLE_DEVICES=3
NUM_WORKERS=4

POSTPROCESSORS=("msp" "odin" "openmax" "mds" "gram" "react" "klm" "vim" "knn" "sirc_ml")

CHECKPOINTS=(
    "./ResNet18-Cifar100/SURE+/best_acc_net_ema_1.pth"
)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


for post in "${POSTPROCESSORS[@]}"; do
    for ckpt in "${CHECKPOINTS[@]}"; do

        parent_dir=$(basename "$(dirname "${ckpt}")")

        ckpt_name=$(basename "${ckpt}")

        num=$(echo "${ckpt_name}" | grep -oP '\d+(?=\.pth)' | tail -n1)
        s_id="s$((num-1))"   # orig_1/ema_1/swa_1 → s0, orig_2/ema_2/swa_2 → s1, ...

        OUTPUT_DIR="output/${parent_dir}/${post}/${s_id}"
        mkdir -p "${OUTPUT_DIR}"

        echo "Running postprocessor: ${post}, checkpoint: ${ckpt}, output_dir: ${OUTPUT_DIR}"

        PYTHONPATH='.':$PYTHONPATH \
        python openood/main.py \
            --config openood/configs/datasets/cifar100/cifar100.yml \
            openood/configs/datasets/cifar100/cifar100_ood.yml \
            openood/configs/networks/resnet18_32x32.yml \
            openood/configs/pipelines/test/test_ood.yml \
            openood/configs/preprocessors/base_preprocessor.yml \
            openood/configs/postprocessors/${post}.yml \
            --num_workers ${NUM_WORKERS} \
            --network.checkpoint "${ckpt}" \
            --network.name resnet18_32x32 \
            --output_dir "${OUTPUT_DIR}"
    done
done