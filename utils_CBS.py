import argparse
import torch
import numpy as np
import pylab as plt
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
# =====================================
# helpers
import cv2, glob, re
from model import transformer_net
# import dabnet
import torch
import tqdm
import argparse
import pandas as pd


@torch.no_grad()
# N.B.: it works with torch.float32 tensor only
def get_styled_image(style_model, image):
    styled = style_model(image)

    # make same size
    styled = F.interpolate(styled, image.size()[2:], mode='bilinear', align_corners=False)

    img = styled[0].cpu().detach().clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return img


def create_style_model(style_number=0):
    model_file = glob.glob('../model/styles/*.pth')[style_number]
    transformer = transformer_net.TransformerNet()
    # load model
    state_dict = torch.load(model_file)

    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    transformer.load_state_dict(state_dict)
    # transformer.to(0)
    transformer.eval()
    return transformer


@torch.no_grad()
def get_masked_image(label, image, category, bg=0):
    # print("image -> type: ", type(image), " shape: ", image.shape, "\n")

    # with torch.no_grad():
    #     input_var = Variable(torch.FloatTensor(image.numpy().copy())).cuda()

    # output = model(input_var)
    # torch.cuda.synchronize()
    # output = label.cpu().data[0].numpy()
    # output = label.transpose(1, 2, 0)
    # output = np.asarray(np.argmax(label, axis=2), dtype=np.uint8)

    output = label[:, :, 0]

    bin_mask = (output == category).astype('uint8')
    if bg:
        bin_mask = 1 - bin_mask

    # masked = bin_mask[:, :, None] * image_original
    masked = bin_mask[:, :, None] * image

    # image = Image.fromarray(masked.astype('uint8'))

    return masked


def save_image(fname, image):
    image = Image.fromarray(image.astype('uint8'))
    image.save(fname)
