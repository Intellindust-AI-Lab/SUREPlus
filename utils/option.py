import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Failure Prediction Framework',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ============================================================
    # 🌍 General Training Settings
    # ============================================================
    parser.add_argument('--epochs', default=20, type=int,
                        help='Total number of training epochs.')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Mini-batch size used during training.')
    parser.add_argument('--nb-run', default=1, type=int,
                        help='Number of repeated runs to compute mean/std of metrics.')
    parser.add_argument('--save-dir', default='./output', type=str,
                        help='Output directory for model checkpoints and logs.')

    # ============================================================
    # ⚙️ Optimizer & Learning Rate Scheduler
    # ============================================================
    parser.add_argument('--lr', default=0.05, type=float,
                        help='Initial learning rate (max value for cosine scheduler).')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='Weight decay coefficient for regularization.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum factor for SGD optimizer.')
    parser.add_argument('--optim-name', default='fsam', type=str,
                        choices=['baseline', 'sam', 'swa', 'fmfp', 'fsam', 'fmfpfsam'],
                        help='Optimization algorithm to use.')

    # --- SAM / FSAM parameters ---
    parser.add_argument('--sigma', default=1.0, type=float, metavar='S',
                        help='Sigma value used in FriendlySAM.')
    parser.add_argument('--lmbda', default=0.8, type=float, metavar='L',
                        help='Lambda coefficient for balancing gradients in FriendlySAM.')
    parser.add_argument('--rho', default=0.2, type=float, metavar='RHO',
                        help='Rho parameter for SAM perturbation radius.')

    # ============================================================
    # 🧮 Loss Function Settings
    # ============================================================
    parser.add_argument('--label-smoothing', default=0.0, type=float,
                        help='Label smoothing factor for cross-entropy loss.')

    # ============================================================
    # 🔁 EMA (Exponential Moving Average)
    # ============================================================
    parser.add_argument('--rebn', action='store_true', default=False,
                        help='If True, use ReBN (Rebalanced BatchNorm) for EMA updates.')

    # ============================================================
    # 🎨 Data Augmentation (PixMix, RegMixup)
    # ============================================================

    # --- PixMix augmentation ---
    parser.add_argument('--pixmix-weight', default=0.0, type=float,
                        help='Loss weight for PixMix augmentation term.')
    parser.add_argument('--pixmix-path', default='/data6022/PixMixSet/fractals_and_fvis/first_layers_resized256_onevis/',
                        type=str,
                        help='Path to the PixMix mixing image dataset (e.g., fractals, feature visualizations).')

    # --- RegMixup augmentation ---
    parser.add_argument('--regmixup-weight', default=0.0, type=float,
                        help='Loss weight for RegMixup regularization term.')
    parser.add_argument('--mixup-beta', default=10.0, type=float,
                        help='Beta distribution parameter for Mixup interpolation.')

    # ============================================================
    # ⚖️ CRL (Consistency Regularization Loss)
    # ============================================================
    parser.add_argument('--crl-weight', default=0.0, type=float,
                        help='Loss weight for consistency regularization loss (CRL).')

    # ============================================================
    # 🧠 Model Configuration
    # ============================================================
    parser.add_argument('--model-name', default='resnet18', type=str,
                        help='Model backbone name (e.g., resnet50, vit_b16, dinov3).')
    parser.add_argument('--use-cosine', action='store_true', default=False,
                        help='Use cosine classifier instead of linear classifier.')
    parser.add_argument('--cos-temp', default=8, type=int,
                        help='Temperature scaling factor for cosine classifier.')

    # --- DINOv3 settings ---
    parser.add_argument('--dinov3-repo', default='/data1032/liyang/git/dinov3', type=str,
                        help='Path to the official DINOv3 repository.')
    parser.add_argument('--dinov3-path', default='/data9022/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth', type=str,
                        help='Path to pretrained DINOv3 weights.')

    # ============================================================
    # 🔁 SWA (Stochastic Weight Averaging)
    # ============================================================
    parser.add_argument('--per_epoch_scheduler', action='store_true', default=False,
                        help='If True, apply scheduler per epoch; otherwise, per iteration.')
    parser.add_argument('--swa-lr', default=0.05, type=float,
                        help='Learning rate used during SWA phase.')
    parser.add_argument('--swa-epoch-start', default=120, type=int,
                        help='Epoch number to start SWA averaging.')

    # ============================================================
    # 🧩 Hardware Configuration
    # ============================================================
    parser.add_argument('--gpu', default=[2, 3, 4, 5], type=int, nargs='+',
                        help='List of GPU device IDs to use for training.')
    parser.add_argument('--nb-worker', default=4, type=int,
                        help='Number of dataloader worker threads.')

    # ============================================================
    # 📚 Dataset Configuration (Subparsers)
    # ============================================================
    subparsers = parser.add_subparsers(title="Dataset Setting", dest="subcommand")

    # --- CIFAR-100 dataset ---
    Cifar100 = subparsers.add_parser("Cifar100",
                                    description='Dataset parser for training on CIFAR-100.',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Configuration for CIFAR-100 dataset.")
    Cifar100.add_argument('--data-name', default='cifar100', type=str, help='Dataset name (used for logging and configs).')
    Cifar100.add_argument("--train-dir", default='/data9022/openood/ours/cifar100/train', type=str, help="Directory containing CIFAR-100 training images.")
    Cifar100.add_argument("--val-dir", default='/data9022/openood/ours/cifar100/val', type=str,  help="Directory containing CIFAR-100 validation images.")
    Cifar100.add_argument("--test-dir", default='/data9022/openood/ours/cifar100/test', type=str, help="Directory containing CIFAR-100 test images.")
    Cifar100.add_argument("--nb-cls", default=100, type=int, help="Number of classes in CIFAR-100.")
    Cifar100.add_argument("--imb-factor", default=1.0, type=float, help="Imbalance factor for simulating long-tailed distribution.")

    # --- ImageNet-1K dataset ---
    ImgNet1k = subparsers.add_parser("ImageNet1k",
                                    description='Dataset parser for training on ImageNet-1K.',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Configuration for ImageNet-1K dataset.")
    ImgNet1k.add_argument('--data-name', default='imagenet1k', type=str, help='Dataset name (used for logging and configs).')
    ImgNet1k.add_argument("--train-dir", default='/data6022/Inet1K/train', type=str, help="Directory containing ImageNet-1K training images.")
    ImgNet1k.add_argument("--val-dir", default='/data6022/Inet1K/val', type=str, help="Directory containing ImageNet-1K validation images.")
    ImgNet1k.add_argument("--test-dir", default='/data6022/Inet1K/val', type=str, help="Directory containing ImageNet-1K test images.")
    ImgNet1k.add_argument("--nb-cls", default=1000, type=int, help="Number of classes in ImageNet-1K.")
    ImgNet1k.add_argument("--imb-factor", default=1.0, type=float, help="Imbalance factor for simulating long-tailed distribution.")

    # ============================================================
    # ✅ Return parsed arguments
    # ============================================================
    return parser.parse_args()
