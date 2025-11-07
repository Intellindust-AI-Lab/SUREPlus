python3 main.py \
--gpu 0 \
--nb-run 3 \
--lr 0.05  \
--weight-decay 0.0005 \
--batch-size 128 \
--epochs 200 \
--model-name resnet18 \
--optim-name fsam \
--regmixup-weight 1.0 \
--rebn \
--save-dir ./ResNet18-Cifar100/SURE-CSC-CRL-SWA+EMA+FSAM \
Cifar100


