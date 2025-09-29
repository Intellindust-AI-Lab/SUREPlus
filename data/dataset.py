import numpy as np
import torch
import uuid
import random
import torch.distributed as dist
import torchvision.transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
import utils.pixmix_utils as pixmix_utils
from copy import deepcopy
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from data.sampler import InfiniteSampler

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2 ** 32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

# Imagent1k
to_tensor = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])   


def augment_input(image):
    aug_list = pixmix_utils.augmentations_all 
    op = np.random.choice(aug_list)
    return op(image.copy(), 3)

def pixmix(orig, mixing_pic, preprocess):
  
    mixings = pixmix_utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)

    for _ in range(np.random.randint(1, 4 + 1)):

        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig))
        else:
            aug_image_copy = tensorize(mixing_pic)

    mixed_op = np.random.choice(mixings)
    mixed = mixed_op(mixed, aug_image_copy, 3)
    mixed = torch.clip(mixed, 0, 1)

    return normalize(mixed)

def generate_imbalanced_data(indices, labels, imb_factor, num_classes):
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    img_max = len(indices) / num_classes
    img_num_per_cls = [int(img_max * (imb_factor ** (i / (num_classes - 1.0)))) for i in range(num_classes)]

    imbalanced_indices = []
    for cls_idx, img_num in zip(range(num_classes), img_num_per_cls):
        cls_indices = class_indices[cls_idx]
        np.random.shuffle(cls_indices)
        selec_indices = cls_indices[:img_num]
        imbalanced_indices.extend(selec_indices)

    return imbalanced_indices

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, is_train=False, dataset_type='normal', imb_factor=None, args=None):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.is_train = is_train
        self.num_classes = 1000 
        if self.is_train and dataset_type in ['cifar10_LT', 'cifar100_LT'] and imb_factor is not None:
            self.gen_imbalanced_data(imb_factor, dataset_type)
        if args:
            self.loss = args.loss

    def gen_imbalanced_data(self, imb_factor, dataset_type):
        targets_np = np.array(self.targets, dtype=np.int64)

        if dataset_type in ['cifar10','cifar10_LT']:
            num_classes = 10
        elif dataset_type in ['cifar100','cifar100_LT']:
            num_classes = 100
        assert dataset_type in ['cifar10','cifar10_LT','cifar100','cifar100_LT'], "error: new dataset should set num_classes"
        imbalanced_indices = generate_imbalanced_data(np.arange(len(self.samples)), targets_np, imb_factor, num_classes)
        self.samples = [self.samples[i] for i in imbalanced_indices]
        self.targets = [self.targets[i] for i in imbalanced_indices]

    def __getitem__(self, index):
        img, label = super(CustomImageFolder, self).__getitem__(index)

        return img, label, index

class MixImageFolder(ImageFolder):
    def __init__(self, root, transform=None, mixing_set=None, target_transform=None, is_train=False, dataset_type='normal', imb_factor=None, args=None):
        super(MixImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.is_train = is_train
        self.mixing_set = mixing_set
        self.num_classes = 1000
        self.preprocess = {'normalize': torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3), 'tensorize': torchvision.transforms.ToTensor()}
        if self.is_train and dataset_type in ['cifar10_LT', 'cifar100_LT'] and imb_factor is not None:
            self.gen_imbalanced_data(imb_factor, dataset_type)
        if args:
            self.loss = args.loss

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.to_tensor = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean, std)
                                        ])   
        
    def gen_imbalanced_data(self, imb_factor, dataset_type):
        targets_np = np.array(self.targets, dtype=np.int64)

        if dataset_type in ['cifar10','cifar10_LT']:
            num_classes = 10
        elif dataset_type in ['cifar100','cifar100_LT']:
            num_classes = 100
        elif dataset_type in ['imagenet1k']:
            num_classes = 1000
        assert dataset_type in ['cifar10','cifar10_LT','cifar100','cifar100_LT'], "error: new dataset should set num_classes"
        imbalanced_indices = generate_imbalanced_data(np.arange(len(self.samples)), targets_np, imb_factor, num_classes)
        self.samples = [self.samples[i] for i in imbalanced_indices]
        self.targets = [self.targets[i] for i in imbalanced_indices]
    

    def __getitem__(self, index):
        img, label = super(MixImageFolder, self).__getitem__(index)
        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        # ce: return label
        # bce: return label_onehot
        # label_onehot = torch.zeros(self.num_classes, dtype=torch.float32).scatter_(0, torch.tensor(label), 1.0)

        return self.to_tensor(deepcopy(img)), pixmix(img, mixing_pic, self.preprocess), label, index

def TrainDataLoader(img_dir, transform_train, batch_size, is_train=True, dataset_type='normal', imb_factor=None, gpu='0'):
    train_set = CustomImageFolder(img_dir, transform_train, is_train=is_train, dataset_type=dataset_type, imb_factor=imb_factor)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    sampler = DistributedSampler(train_set, shuffle=True)

    batch_sampler = BatchSampler(sampler, batch_size // len(gpu), drop_last=True)
    dataloader_kwargs = {"num_workers": 4, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
    train_loader = DataLoader(train_set, **dataloader_kwargs)
    return train_loader

def TrainMixDataLoader(img_dir, transform_train, batch_size, mixing_set, is_train=True, dataset_type='normal', imb_factor=None, gpu=[0], args=None):
    train_set = MixImageFolder(img_dir, transform_train, mixing_set=mixing_set, is_train=is_train, dataset_type=dataset_type, imb_factor=imb_factor, args=args)
    dataloader_kwargs = {
        "num_workers": 8,
        "pin_memory": True,
        "worker_init_fn": worker_init_reset_seed
    }
    if args.gpu is not None and dist.is_initialized():
        sampler = DistributedSampler(train_set, shuffle=True)
        batch_sampler = BatchSampler(sampler, batch_size // len(gpu), drop_last=True)
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(train_set, **dataloader_kwargs)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **dataloader_kwargs)

    return train_loader

# test data loader
def TestDataLoader(img_dir, transform_test, batch_size, dataset_type, args):
    test_set = CustomImageFolder(img_dir, transform_test, dataset_type=dataset_type, args=args)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    return test_loader

def get_loader(args, dataset, train_dir, val_dir, test_dir, batch_size, imb_factor, model_name, gpu=[0]):

    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    nb_cls = 1000

    # ResNet50
    train_aug_list = [
        torchvision.transforms.Resize(256, interpolation = InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        # torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(norm_mean, norm_std)
        ] 
    test_aug_list = [
        torchvision.transforms.Resize(256, interpolation = InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(norm_mean, norm_std)]


        
    # transformation of the train set
    transform_train = torchvision.transforms.Compose(train_aug_list)
    # transformation of the test set
    transform_test = torchvision.transforms.Compose(test_aug_list)

    mixing_set_transform = torchvision.transforms.Compose(
                                                    [torchvision.transforms.Resize(256), 
                                                    torchvision.transforms.RandomCrop(224)])  
      
    mixing_set = ImageFolder(args.pixmix_path, transform=mixing_set_transform)

    # PixMix
    train_loader = TrainMixDataLoader(train_dir, transform_train, batch_size, mixing_set, is_train=True, dataset_type=dataset, imb_factor=imb_factor, gpu=gpu, args=args)
    # Normal
    # train_loader = TrainDataLoader(train_dir, transform_train, batch_size, is_train=True, dataset_type=dataset, imb_factor=imb_factor)

    val_loader = TestDataLoader(val_dir, transform_test, 16, dataset_type=dataset, args=args)
    test_loader = TestDataLoader(test_dir, transform_test, 16, dataset_type=dataset, args=args)

    return train_loader, val_loader, test_loader, nb_cls

