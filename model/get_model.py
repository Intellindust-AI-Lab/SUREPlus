import model.resnet18
import model.classifier
import torch



def get_model(model_name, nb_cls, logger, args):
    if model_name == 'resnet18':
        net = model.resnet18.ResNet18(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name =="dinov3_l16":
        net = torch.hub.load(args.dinov3_repo, 'dinov3_vitl16', source='local', weights=args.dinov3_path)
         # Remove the original classification head
        if args.use_cosine:
            # Cosine classifier (e.g. for open-set / OOD tasks)
            net.head = model.classifier.Classifier(1024, nb_cls, args.cos_temp).cuda()
        else:
            # Standard linear classifier
            net.head = torch.nn.Linear(1024, nb_cls).cuda()
    
    

    return net
