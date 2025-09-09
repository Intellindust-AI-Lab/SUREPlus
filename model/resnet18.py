import torch
import torch.nn as nn
import torch.nn.functional as F
import model.classifier

def get_activation(name):
    if name == 'relu':
        return F.relu
    elif name == 'gelu':
        return F.gelu
    elif name == 'lrelu':
        return lambda x: F.leaky_relu(x, negative_slope=0.01)  # default slope is 0.01
    else:
        raise ValueError(f"Unsupported activation: {name}")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation='relu'):
        super(BasicBlock, self).__init__()
        self.act = get_activation(activation)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation='relu'):
        super(Bottleneck, self).__init__()
        self.act = get_activation(activation)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, use_cos=False, cos_temp=8, activation='relu'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_cos = use_cos
        self.mc_dropout = False
        self.act = get_activation(activation)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, activation=activation)

        if self.use_cos : 
            self.classifier = model.classifier.Classifier(512 * block.expansion, num_classes, cos_temp)
        else : 
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride, activation='relu'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature_output=False):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        dim = out.size(-1)
        out = F.avg_pool2d(out, dim)
        out = out.view(out.size(0), -1)
        y = self.classifier(out) if self.use_cos else self.linear(out)
        return (y, out) if feature_output else y


def ResNet18(num_classes, use_cos=False, cos_temp=8, activation='relu'):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_cos, cos_temp, activation)
