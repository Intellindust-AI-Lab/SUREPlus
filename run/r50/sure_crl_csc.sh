## Ours
python3 main.py \
--gpu 6 7 8 9 \
--nb-run 1 \
--lr 0.001 \
--model-name resnet50 \
--optim-name fmfp \
--mixup-weight 1.0 \
--mixup-beta 10 \
--save-dir ./ResNet50-ImageNet1k/sure-crl-csc \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val


