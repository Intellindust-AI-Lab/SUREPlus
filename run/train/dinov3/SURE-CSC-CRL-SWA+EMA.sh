python3 main.py \
--gpu 0 1 2 3 4 5 \
--nb-run 1 \
--lr 0.00001 \
--weight-decay 0.000005 \
--batch-size 64 \
--epochs 20 \
--model-name dinov3_l16 \
--optim-name sam \
--regmixup-weight 1.0 \
--mixup-beta 10.0 \
--dinov3-repo /data1032/liyang/git/dinov3 \
--dinov3-path /data9022/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
--save-dir ./DinoV3_L16-ImageNet1k/SURE-CSC-CRL-SWA+EMA-ReBN \
ImageNet1k \
--train-dir /data6022/Inet1K/train \
--val-dir /data6022/liyang/git/surev2-inet/imagenet/val


