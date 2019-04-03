import argparse

import torch
import torch.nn as nn
from torchvision import models

from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# noinspection PyShadowingNames
def dream_deep(model, image):
    print(model)
    print(image)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='images/')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_octaves', type=int, default=10, help='number of octaves')
    parser.add_argument('--octave_scale', type=float, default=1.4, help='image scale between octaves')
    parser.add_argument('--num_iterations', type=int, default=20, help='number of iterations for gradient ascent')
    parser.add_argument('--layer', type=int, default=20, help='layer at which to perform deep dream')
    args = parser.parse_args()

    in_image = Image.open(args.input_image)

    model = models.vgg19(pretrained=True)
    layers = list(model.features.children())
    model = nn.Sequential(*layers[: (args.layer + 1)])
    model.to(device=device)

    dream_deep(model, in_image)
