import copy
import os.path
import time
import numpy as np
import torch
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader,Dataset, random_split
from torchvision import transforms
from torch.backends import cudnn
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace

### FedTracker utils 
# The models used in this challenge are based on the FedTracker framework, 
# but do not accurately reflect the original FedTracker implementation.
# The white-box fingerprinting method was simplified for educational purposes.
# The original FedTracker code can be found at: https://github.com/shaoshuo-ss/FedTracker

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(outchannel)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(outchannel))
        ]))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1,1],padding=[0,1,0],first=False) -> None:
        super(Bottleneck,self).__init__()
        self.bottleneck = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False)),
            ('bn3', nn.BatchNorm2d(out_channels*4))
        ]))

        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels*4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, args):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, args.num_classes)
        self.memory = dict()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = OrderedDict()
        count = 1
        for stride in strides:
            name = "layer{}".format(count)
            layers[name] = block(self.in_channel, channels, stride)
            self.in_channel = channels
            count += 1
        return nn.Sequential(layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            # self.memory = dict()
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)

def ResNet18(args):
    return ResNet(ResidualBlock, args)

def get_model(args):
    if args.model == 'ResNet18':
        return ResNet18(args)
    else:
        exit("Unknown Model!")

def get_full_dataset(dataset_name, img_size=(32, 32)):
    if dataset_name == 'cifar10':
        train_dataset = CIFAR10('./oracle/data/cifar10/', train=True, download=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Pad(4, padding_mode="reflect"),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                                ]))
        test_dataset = CIFAR10('./oracle/data/cifar10/', train=False, download=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                                   transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                               ]))
    else:
        exit("Unknown Dataset")
    return train_dataset, test_dataset

def get_embed_layers(model, embed_layer_names):
    embed_layers = []
    embed_layer_names = embed_layer_names.split(";")
    for embed_layer_name in embed_layer_names:
        embed_layer = model
        for name in embed_layer_name.split('.'):
            embed_layer = getattr(embed_layer, name)
        embed_layers.append(embed_layer)
    return embed_layers


### Arguments
args = None
def initialize_environment():
    with open("./models/args.txt", "r") as f:
        global args
        args = Namespace(**json.load(f))

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

### Helper functions
def load_model():
    model = get_model(args)
    model.load_state_dict(torch.load("./models/model_9.pth", map_location=args.device))
    return model

def oracle(model):
    model.eval()
    model.to(args.device)
    extracting_matrices = np.load("./oracle/extracting_matrices_ResNet18_D4S.npy", allow_pickle=True)
    local_fingerprints = np.load("./oracle/local_fingerprints_ResNet18_D4S.npy", allow_pickle=True)    
    embed_layers = get_embed_layers(model, args.embed_layer_names)
    test_dataset = get_full_dataset(args.dataset)[1]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    min_ber = 100000
    min_idx = 0
    bit_length = local_fingerprints[0].shape[0]
    model.to("cpu")
    weight = embed_layers[0].weight.detach().numpy()
    for i in range(1, len(embed_layers)):
        weight = np.append(weight, embed_layers[i].weight.detach().numpy())
    for idx in range(len(local_fingerprints)):
        matrix = extracting_matrices[idx]
        result = np.dot(matrix, weight)
        result[result >= 0] = 1
        result[result < 0] = -1
        ber = np.sum(result != local_fingerprints[idx]) / bit_length
        if ber < min_ber:
            min_ber = ber
            min_idx = idx
    print("* Test Accuracy: {:.2f}%".format(acc * 100))
    if min_ber <= 0.2301:
        print("* Watermark Detected!")
        print("  - Participant ID: {}".format(min_idx))
        print("  - Bit Error Rate: {:.4f}".format(min_ber))
    else:
        print("* Watermark Not Detected...")
    return

def plot_bn_distributions(model):
    plt.figure(figsize=(12, 5))
    # Collect all BN scaling weights into a dataframe or clean format for seaborn
    data_list = []
    for name, module in model.named_modules():
        if "bn" in name and module.weight is not None:
            weights = module.weight.detach().cpu().numpy()
            for w in weights:
                data_list.append({"Layer": name, "Scaling Value": w})      
    df = pd.DataFrame(data_list)
    if df.empty:
        print("No BatchNorm weights found to plot!")
        return
    # Plot using Seaborn KDE (smooth estimated distribution)
    ax = sns.kdeplot(
        data=df, 
        x="Scaling Value", 
        hue="Layer", 
        common_norm=False, 
        palette="gist_rainbow", 
        alpha=0.4, 
        fill=True, 
        linewidth=1.5
    )
    plt.xlim(-0.5, 1.5)
    plt.ylim(0, 10)
    plt.xlabel('Scaling Value ($\gamma$)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero')
    plt.axvline(x=1, color='k', linestyle='-', alpha=0.5, linewidth=1.5, label='Unity (Initialization)')
    plt.annotate(
        'Original Initialization', 
        xy=(1.0, 5),          # Point the arrow tip touches (x=1, y=50% up the plot)
        xytext=(1.15, 5),     # Where the text label is positioned
        va='center',
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
        fontsize=10, 
        color='black'
    )
    
    plt.title('BN Scaling Factor Distributions (KDE)', fontsize=14, fontweight='bold')
    # Clean up the legend outside the plot
    sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=True)    
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    return

def read_bn_scaling(model,layer_name):
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            return module.weight.detach().cpu().numpy()
    raise ValueError(f"Layer {layer_name} not found or does not have a weight attribute.")

def write_bn_scaling(model, layer_name, vector):
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.copy_(torch.tensor(vector, device=module.weight.device))
            return
    raise ValueError(f"Layer {layer_name} not found or does not have a weight attribute.")

def reset_bn_scaling(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.fill_(1.0)
            return
    raise ValueError(f"Layer {layer_name} not found or does not have a weight attribute.")

def load_participant_model(id):
    model_path = f"./models/model_{id}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"You don't have access to {id}'s model'.")
    model = get_model(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    return model

def compare_models(model1, model2):
    for name, param1 in model1.named_parameters():
        param2 = dict(model2.named_parameters())[name]
        if not torch.equal(param1, param2):
            print("Models differ at layer:", name)
    return