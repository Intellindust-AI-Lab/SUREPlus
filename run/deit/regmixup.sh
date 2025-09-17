## Ours
python3 main.py \
--gpu 2 3 6 7 8 4 \
--nb-run 1 \
--lr 0.0001 \
--weight-decay 0.00005 \
--epochs 20 \
--model-name deit_base_patch16_224 \
--optim-name baseline \
--pixmix-weight 0.0 \
--mixup-weight 1.0 \
--mixup-beta 10.0 \
--save-dir ./Deit-B-ImageNet1k/regmixup \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val


