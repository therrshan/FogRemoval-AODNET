import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
from model.dataloader import *
from model.net import *
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


def dehaze_image(image_path):

	data_hazy = Image.open(image_path)
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.unsqueeze(0)

	dnet = dehaze_net()
	dnet.load_state_dict(torch.load('model/dehazer.pth', map_location=torch.device('cpu')))

	clean_image = dnet(data_hazy)
	torchvision.utils.save_image(clean_image, "static/results/" + image_path.split("/")[-1])
	return torchvision.utils.make_grid(torch.cat((data_hazy, clean_image),0))

