python3 main.py \
--gpu 1 2 3 4 6 7 8 9 \
--nb-run 1 \
--lr 0.00001 \
--weight-decay 0.000005 \
--batch-size 64 \
--epochs 20 \
--model-name dinov3_l16 \
--optim-name fmfp \
--mixup-weight 1.0 \
--mixup-beta 10.0 \
--crl-weight 1.0 \
--use-cosine \
--dinov3-repo /data1032/liyang/git/dinov3 \
--dinov3-path /data/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
--save-dir ./DinoV3_L16-ImageNet1k/SURE \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir imagenet/val


