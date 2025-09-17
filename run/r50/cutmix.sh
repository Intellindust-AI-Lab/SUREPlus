## cutmix
python3 main.py \
--gpu 4 5 6 7 \
--nb-run 1 \
--model-name resnet50 \
--optim-name baseline \
--cutmix-weight 1.0 \
--save-dir ./ResNet50-ImageNet1k/CutMix \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val 
