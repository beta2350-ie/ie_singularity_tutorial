import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def give_color(img, n_classes):
    colors = sns.color_palette(n_colors=n_classes)
    color_mask = np.zeros((img.shape[0], img.shape[1], 3))
    for c in range(n_classes):
        c_bool = (img == c)
        color_mask[:, :, 0] += (c_bool * colors[c][0])
        color_mask[:, :, 1] += (c_bool * colors[c][1])
        color_mask[:, :, 2] += (c_bool * colors[c][2])
    return color_mask

def main():
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    # or
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
    model.eval()

    #url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
    #try: urllib.URLopener().retrieve(url, filename)
    #except: urllib.request.urlretrieve(url, filename)

    path = "cat.4001.jpg"
    # sample execution (requires torchvision)
    input_image = Image.open(path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    mask = output_predictions.cpu().detach().numpy()
    mask.shape
    color_mask = give_color(mask, 21)
    plt.imshow(color_mask);

    origin = Image.open(path)
    out = Image.fromarray(np.uint8(color_mask*255))
    mask = Image.new('L', origin.size, 128)

    im = Image.composite(origin, out, mask)
    im.save("segmentation_image.png")

if __name__=="__main__":
    main()
