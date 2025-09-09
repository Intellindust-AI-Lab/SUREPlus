## Baseline
python3 main.py \
--gpu 4 5 6 7 \
--nb-run 1 \
--model-name resnet50 \
--optim-name baseline \
--save-dir ./ResNet50-ImageNet1k/CE \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val 
