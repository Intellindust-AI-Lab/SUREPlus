# ## SURE
# python3 main.py \
# --gpu 6 7 8 9 \
# --nb-run 1 \
# --lr 0.005 \
# --model-name resnet50 \
# --optim-name fmfp \
# --use-cosine \
# --mixup-weight 1.0 \
# --mixup-beta 10 \
# --crl-weight 1.0 \
# --save-dir ./ResNet50-ImageNet1k/SURE_lr0.005 \
# ImageNet1k \
# --train-dir /data6022/Inet1K/train \
# --val-dir imagenet/val


## Ours
python3 main.py \
--gpu 6 7 8 9 \
--nb-run 1 \
--lr 0.001 \
--model-name resnet50 \
--optim-name sam \
--pixmix-weight 0.0 \
--mixup-weight 1.0 \
--mixup-beta 10 \
--rebn \
--save-dir ./ResNet50-ImageNet1k/sure-crl-csc-swa+ema \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val



# python3 main.py \
# --gpu 6 7 8 9 \
# --nb-run 1 \
# --lr 0.001 \
# --model-name resnet50 \
# --optim-name fsam \
# --pixmix-weight 1.0 \
# --mixup-weight 1.0 \
# --mixup-beta 10 \
# --rebn \
# --save-dir ./ResNet50-ImageNet1k/sure-crl-csc-swa+ema+fsam \
# ImageNet1k \
# --train-dir /data6022/Inet1K/train \
# --val-dir imagenet/val
