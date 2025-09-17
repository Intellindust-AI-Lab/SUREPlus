## Baseline
python3 main.py \
--gpu 0 1 2 3  \
--nb-run 1 \
--batch-size 64 \
--model-name resnet50 \
--optim-name baseline \
--save-dir ./ResNet50-ImageNet1k/CE \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val 
