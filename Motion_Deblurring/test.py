## Restormer: Efficient Transformer for High-Resolution Image Restoration
## https://arxiv.org/abs/2111.09881

import sys
sys.path.append(r"E:/Restormer")

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')
parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default=r"E:/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth", type=str, help='Path to weights')
args = parser.parse_args()

####### Load yaml #######
yaml_file = r"E:/Restormer/Motion_Deblurring/Options/Deblurring_Restormer.yml"
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===> Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


datasets = ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
factor = 8

for dataset in datasets:
    print(f"\n===> Processing Dataset: {dataset}")
    result_dir = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    inp_dir = os.path.join(args.input_dir, dataset)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_)) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)

            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
                           img_as_ubyte(restored))

    print(f"✅ {dataset} 저장 완료: {os.path.abspath(result_dir)}")

print("\n✅ 모든 복원 이미지 저장 완료.")

# (base) PS E:\Restormer\Motion_Deblurring> python test.py --input_dir E:/Restormer/Motion_Deblurring/Datasets