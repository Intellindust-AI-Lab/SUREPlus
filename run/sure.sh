## SURE
python3 main.py \
--gpu 6 7 8 9 \
--nb-run 1 \
--model-name resnet50 \
--optim-name fmfp \
--use-cosine \
--mixup-weight 1.0 \
--mixup-beta 10 \
--crl-weight 1.0 \
--save-dir ./ResNet50-ImageNet1k/SURE \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val
