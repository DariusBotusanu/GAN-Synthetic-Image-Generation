import numpy as np
import os
from PIL import Image

import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

def convert_image_to_shape(image_path, target_shape):
    # Open the image from the specified path
    img = Image.fromarray(np.load(image_path))

    # Convert the PIL image to grayscale if the target shape specifies a single channel
    if img.mode != 'L' and target_shape[0] == 1:
        img = img.convert('L')

    # Resize the image to the target shape (width, height)
    img_resized = np.array(img.resize((target_shape[2], target_shape[1]), Image.LANCZOS))

    return img_resized.reshape((1, target_shape[2], target_shape[1]))/255

def show_tensor_images(image_tensor, num_images=25, size=(1, 126, 126)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    #image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def get_images(folder_path='drive/MyDrive/Giraffa Analytics/MRI-Blur-Detection/Data/preprocessed slices', img_shape=(1,126,126)):
    images = []
    for img_path in os.listdir(folder_path):
        if  ("nomotion" in img_path) and int(img_path.split('_')[1]) >= 110 and int(img_path.split('_')[1]) <=130:
          image = convert_image_to_shape(f'{folder_path}/{img_path}', target_shape=img_shape)
          images.append(torch.tensor(image).float())
    return images
    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]