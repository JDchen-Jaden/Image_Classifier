import argparse
import torch
import torchvision
from collections import OrderedDict
from torch import nn, optim
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import os, random
import matplotlib.pyplot as plt
import json
import seaborn as sns
from workspace_utils import keep_awake



# enable command line interface
def cli():
    """
    Apply argharse to allow command line interface
    """
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('data_dir', type = str, default = 'flowers', help ='Directory of Image')
    parser.add_argument('--save_dir', type = str, help ='Directory to save checkpoints')
    parser.add_argument('--arch', action = 'store', dest = 'arch', default = 'vgg16', help = 'NN Model')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type = int, default = 2048, help='Hidden neurons')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Epoch')
    parser.add_argument('--gpu', action  = 'store', default = False, help = 'GPU mode')
    return parser.parse_args()

# Image transforms for training, validation, and testing
def img_transforms(train_dir, valid_dir, test_dir):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(train_dir, transform = train_transforms) 
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms) 
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms) 

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 80, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 80)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 80)
    
    return train_data, valid_data, test_data, train_loader, test_loader, valid_loader

# build classifier
def my_classifier(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        model = models.vgg16(pretrained = True)
        print('vgg16 is applied')
                            
    for param in model.parameters():
        param.requires_grad = False                        
    
    features = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_units, int(hidden_units/2))),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(int(hidden_units/2), 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    return model, classifier

# NN trainer
def trainer(args, model, train_loader, valid_loader, device, optimizer, criterion):
    
    epochs = args.epochs
    train_losses, valid_losses = [], []
    for i in keep_awake(range(5)):
        for epoch in range(epochs):
            running_loss = 0
            for inputs,labels in train_loader:

                inputs,labels = inputs.to(device),labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps,labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            else:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs,labels = inputs.to(device),labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss/len(train_loader))
                    valid_losses.append(valid_loss/len(valid_loader))

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                model.train()
                            
# save checkpoint
def save_checkpoint(model, train_data, args, optimizer):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': model.classifier[0].in_features,
                  'output_size': 102,
                  'pretrained_NN': args.arch,
                  'state_dict': model.state_dict(),
                  'epochs':args.epochs,
                  'learning_rate': args.learning_rate,
                  'classifier': model.classifier,
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint,'checkpoint.pth')

def main():
    args = cli()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'                        
                            
    train_data, valid_data, test_data, train_loader, test_loader, valid_loader = img_transforms(train_dir, valid_dir, test_dir)
                        
    model, classifier = my_classifier(args.arch, args.hidden_units) 
    model.classifier = classifier

                            
    if args.gpu:
        device = torch.device('cuda')
        print('GPU mode is activated')
    else:
        device = torch.device('cpu')

    model.to(device)
                            
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)                        
    
    trainer(args, model, train_loader, valid_loader, device, optimizer, criterion)
                            
    save_checkpoint(model, train_data, args, optimizer)              
    
if __name__ == '__main__': 
    main()
