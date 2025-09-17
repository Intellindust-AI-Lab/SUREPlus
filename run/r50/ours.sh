## Ours
python3 main.py \
--gpu 6 7 8 4 \
--nb-run 1 \
--lr 0.001 \
--weight-decay 0.0005 \
--model-name resnet50 \
--optim-name baseline \
--pixmix-weight 1.0 \
--mixup-weight 1.0 \
--mixup-beta 10.0 \
--rebn \
--save-dir ./ResNet50-ImageNet1k/ours_lr0001_baseline \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val


