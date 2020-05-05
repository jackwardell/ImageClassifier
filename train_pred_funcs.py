
import click
import os
import json
import torch
import time
import PIL
import torchvision
import numpy as np

from collections import OrderedDict
from PIL import Image
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sb


def get_dataloaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(30),
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]
        ),
        'valid': torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]
        ),
        'test': torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]
        )
    }

    image_datasets = {
        'train': torchvision.datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': torchvision.datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': torchvision.datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=42, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=42,shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=42, shuffle=True),
    }
    
    return dataloaders, image_datasets


def get_model(arch="vgg16", learning_rate=0.001, input_features=25088, hidden_units=2048, dropout=0.5, step_size=5, gamma=0.1, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(torchvision.models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_features, hidden_units)),
                      ('relu', nn.ReLU()),
                      ('dropout', nn.Dropout(dropout)),
                      ('fc2', nn.Linear(hidden_units, 102)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))
    
    model.classifier = classifier
    model.arch = arch
    criterion = nn.NLLLoss()
    optimiser = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=gamma)
    model = model.to(device)
    return model, optimiser, criterion, scheduler


def train_model(model, optimiser, criterion, scheduler, dataloaders, image_datasets, epochs=5, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
        for phase in ("train", "valid"):
            if phase == "train":
                scheduler.step()
                model.train() 
                model.to(device)
            else:
                model.eval() 
            running_loss = 0
            running_corrects = 0
           
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
    
                optimiser.zero_grad()
    
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
    
                    if phase == "train":
                        loss.backward()
                        optimiser.step()
                    else:
                        pass
                    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)
                    
            epoch_error = running_loss / len(image_datasets[phase])
            epoch_accuracy = running_corrects.double() / len(image_datasets[phase])
                        
            print(f"Epoch: {epoch}")
            print(f'{phase} epoch error: {epoch_error}')
            print(f'{phase} epoch accuracy: {epoch_accuracy}')
            print('********')
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    return model


def save_checkpoint(model, checkpoint_path='checkpoint.pth'):  
    checkpoint = {
        'structure': model.arch,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    

def load_checkpoint(checkpoint_path='checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    model = getattr(torchvision.models, checkpoint['structure'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model, *_ = get_model()
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def predict(image_file_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()
    model.cpu()
    
    img = process_image(image_file_path)
    img = img.unsqueeze(0)
    img = torch.autograd.Variable(img, volatile=True)
    
    outputs = model.forward(img)
    top_prob, top_labels = torch.topk(outputs, topk)
    top_prob = top_prob.exp().data.numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_labels.data.numpy()[0]]
    
    return top_prob, top_classes


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    img = PIL.Image.open(image)
    img.load()
    
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]
            )
        ]
    )         
    return transform(img)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax