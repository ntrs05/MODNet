# inference.py
import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.models.modnet import MODNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input images or a single image file')
    parser.add_argument('--output-path', type=str, help='path of output images')
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f'Cannot find input path: {args.input_path}')
        exit()
    if not os.path.exists(args.output_path):
        print(f'Cannot find output path: {args.output_path}')
        exit()
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find checkpoint path: {args.ckpt_path}')
        exit()

    ref_size = 512

    im_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(args.ckpt_path)
    else:
        weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))

    modnet.load_state_dict(weights)
    modnet.eval()

    if os.path.isdir(args.input_path):
        im_names = os.listdir(args.input_path)
    else:
        im_names = [os.path.basename(args.input_path)]

    for im_name in im_names:
        full_path = os.path.join(args.input_path, im_name) if os.path.isdir(args.input_path) else args.input_path
        print(f'Processing image: {full_path}')

        im = Image.open(full_path)
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        im = im_transform(Image.fromarray(im))
        im = im[None, :, :, :]

        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()

        original_im = np.asarray(Image.open(full_path))
        if original_im.shape[2] == 4:
            original_im = original_im[:, :, :3]

        foreground = np.zeros((original_im.shape[0], original_im.shape[1], 4), dtype=np.uint8)
        foreground[..., :3] = original_im
        foreground[..., 3] = (matte * 255).astype(np.uint8)

        foreground_name = im_name.split('.')[0] + '_foreground.png'
        Image.fromarray(foreground).save(os.path.join(args.output_path, foreground_name), format='PNG')
        print(f'Saved foreground image: {foreground_name}')