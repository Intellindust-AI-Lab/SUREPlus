## Ours
python3 main.py \
--gpu 0 1 2 3 \
--nb-run 1 \
--model-name resnet50 \
--optim-name fsam \
--pixmix-weight 1.0 \
--mixup-weight 1.0 \
--mixup-beta 10 \
--rebn \
--save-dir ./ResNet50-ImageNet1k/ours \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val
