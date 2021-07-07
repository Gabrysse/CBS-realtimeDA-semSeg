import argparse
import torch
import numpy as np
import pylab as plt
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
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
import random


# image and label from train_DA_V2 come in a shape like (D, H, W)
def class_base_styling(image, label, class_id=[7], style_id=0, loss='crossentropy', j=0):
    # Creation of styling model

    # style_model = create_style_model(style_id)

    if loss == 'dice':
        label = np.argmax(image, axis=0)

    fg_image = get_masked_image(label, image.transpose(1, 2, 0), category=class_id, bg=0)
    bg_image = get_masked_image(label, image.transpose(1, 2, 0), category=class_id, bg=1)
    fg_image = fg_image.transpose(2, 0, 1)
    bg_image = bg_image.transpose(2, 0, 1)

    # save_image("fg_image.png", fg_image)
    # save_image("bg_image.png", bg_image)

    # image = image.transpose(2, 0, 1)
    # image = torch.from_numpy(image.copy())
    # image = transforms.Lambda(lambda x: x.mul(255))(image)

    # if torch.cuda.is_available():
    #     image = image.unsqueeze(0).to(0)
    #     image = image.cuda()
    # else:
    #     image = image.unsqueeze(0)
    # image_style1 = get_styled_image(style_model, image)

    image_style1 = stylize(image)

    # save_image("./images/"+str(j)+"imagestyle.png", image_style1)

    # Apply local style to fg
    # image_style1 = image_style1.transpose(1, 0).transpose(1, 2)

    fg_styled = image_style1 * (fg_image != 0)
    # Apply local style to bg
    # bg_styled = image_style1 * (bg_image != 0)

    output = fg_styled + bg_image

    # save_image("./images/"+str(j)+"final_image.png", output)

    return output

# #####################################################################################################################

@torch.no_grad()
# N.B.: it works with torch.float32 tensor only
def get_styled_image(style_model, image):
    styled = style_model(image).squeeze(0)

    print(styled.shape)
    img = transforms.ToPILImage()(styled)
    img.save("styled.png")

    # make same size
    styled = F.interpolate(styled, image.size()[2:], mode='bilinear', align_corners=False)

    img = styled[0].cpu().detach().clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return img
    # return styled


def create_style_model(style_number=0, path='./model/styles/*.pth'):
    model_file = glob.glob(path)[style_number]
    transformer = transformer_net.TransformerNet()

    # load model
    state_dict = torch.load(model_file)

    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    transformer.load_state_dict(state_dict)

    if torch.cuda.is_available():
        transformer.to(0)

    transformer.eval()
    return transformer


@torch.no_grad()
# image and label must be in the shape as follows: (H, W, D)
def get_masked_image(label, image, category, bg=0):
    # print("image -> type: ", type(image), " shape: ", image.shape, "\n")

    # with torch.no_grad():
    #     input_var = Variable(torch.FloatTensor(image.numpy().copy())).cuda()

    # output = model(input_var)
    # torch.cuda.synchronize()
    # output = label.cpu().data[0].numpy()
    # output = label.transpose(1, 2, 0)
    # output = np.asarray(np.argmax(label, axis=2), dtype=np.uint8)

    output = label[:, :]

    # bin_mask = (output == category).astype('uint8')
    bin_mask = np.isin(output, category).astype('uint8')

    if bg:
        bin_mask = 1 - bin_mask

    masked = (bin_mask[:, :, None] * image) * 255

    img = Image.fromarray(masked.astype('uint8'))
    img.save("./images/" + str(bg) + "_fg-bg.png")

    return masked


def save_image(fname, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(fname)


def stylize(content_image, modelPath='./model/styles/camvid3.pth'):

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.transpose(1, 0).transpose(1, 2)

    if torch.cuda.is_available():
        content_image = content_image.unsqueeze(0).to(0)
    else:
        content_image = content_image.unsqueeze(0)

    with torch.no_grad():
        style_model = transformer_net.TransformerNet()
        state_dict = torch.load(modelPath)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            style_model.to(0)

        output = style_model(content_image).cpu()

    return output[0]
