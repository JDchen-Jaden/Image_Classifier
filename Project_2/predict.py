import argparse
import torch
import torchvision
from torch import nn, optim
from collections import OrderedDict
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import os, random
import json
from workspace_utils import keep_awake
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# enable command line interface
def cli():
    """
    Apply argharse to allow command line interface
    """
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('image_path', type = str, default = './flowers/test/1/image_06743.jpg', help ='Directory of Image of testing')
    parser.add_argument('checkpoint', type = str, default = 'checkpoint.pth', help ='Directory to save checkpoints')
    parser.add_argument('--top_k', action = 'store', dest = 'top_k', type = int, default = 5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action  = 'store', default = False, help = 'GPU mode')
    return parser.parse_args()

# load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['pretrained_NN'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    else: 
        model = models.vgg16(pretrained = True)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model 

    
    
# process image
def process_image(image):
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    image = image_transforms(image)
    
    return image

# predict
def pridict(device, cat_to_name, image_path, model, top_k):
    
    model.to(device)
    model.eval()
    image = process_image(image_path).unsqueeze_(0).float()
    image = image.to(device)

    with torch.no_grad():
        ps = torch.exp(model(image))
        
    idx = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}    
    
    top_ps, top_classes = ps.topk(top_k, dim=1)
    label = [idx[i] for i in top_classes.tolist()[0]]
    top_ps = top_ps.tolist()[0]
    return top_ps, label
    
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

def result_show(image_path, ps, label):
    ax = plt.subplot(2,1,1)

    image = process_image(image_path)

    imshow(image, ax = plt)

    plt.subplot(2,1,2)
    sns.barplot(x=ps, y=label, color=sns.color_palette()[0]);
    plt.show()

def main():
    
    args = cli()
    
    if args.gpu:
        device = torch.device('cuda')
        print('GPU mode is activated')
    else:
        device = torch.device('cpu')
        
    
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    
    model = load_checkpoint(checkpoint)
    
    model = model.to(device)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    
    image = process_image(image_path)
    
    top_ps, label = pridict(device, cat_to_name, image_path, model, top_k)
    
    #imshow(image)
    #result_show(image_path, top_ps, label)
    
    result_dic = {'Labels':label, 'Probability':top_ps}
    results = pd.DataFrame(result_dic)
    print(results)
    
    
if __name__ == '__main__': 
    main()
    