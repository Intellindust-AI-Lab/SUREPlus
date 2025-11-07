python3 main.py \
--gpu 7 \
--nb-run 1 \
--lr 0.05 \
--weight-decay 0.0005 \
--batch-size 128 \
--epochs 200 \
--model-name resnet18 \
--optim-name fmfp \
--regmixup-weight 1.0 \
--crl-weight 1.0 \
--use-cosine \
--cos-temp 8.0 \
--save-dir ./ResNet18-Cifar100/SURE \
Cifar100


