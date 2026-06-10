"""
 @Time    : 2026 Update (Original: 9/29/19)
 @Author  : TaylorMei (Updated for 2-Scale Pyramid Inference & CPU optimization)
 @Project : ICCV2019_MirrorNet
 @File    : infer_pyramid.py
"""
import numpy as np
import os
import cv2

import torch
from PIL import Image
from torchvision import transforms

from .misc import crf_refine
from .mirrornet import MirrorNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_path = './ckpt'
exp_name = 'MirrorNet'
args = {
    'snapshot': '160',
    'scale_low': 384,       # Scala ridotta (Macro specchi)
    'scale_high': 640,      # Scala maggiorata (Micro dettagli e cornici)
    'crf': False            
}

def get_transform(target_scale):
    return transforms.Compose([
        transforms.Resize((target_scale, target_scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

to_pil = transforms.ToPILImage()

def load_mirrornet(weights_path, device):
    """
    Inizializza MirrorNet e carica i pesi. 
    """
    net = MirrorNet().to(device)
    if weights_path and os.path.exists(weights_path):
        print(f'Caricamento pesi MirrorNet da: {weights_path}')
        net.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("ATTENZIONE: Pesi MirrorNet non trovati!")
    
    net.eval()
    return net

def infer_mirrornet(img_path, mirrornet_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if mirrornet_weights is not None:
        net = load_mirrornet(mirrornet_weights, device)
    else:
        return -1

    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    w, h = img.size
    to_pil = transforms.ToPILImage()

    transform_low = get_transform(args['scale_low'])
    transform_high = get_transform(args['scale_high'])

    with torch.no_grad():
        img_var_low = transform_low(img).unsqueeze(0).to(device)
        _, _, _, f_1_low = net(img_var_low)
        f_1_low = f_1_low.detach().squeeze(0).cpu()
        mask_low = np.array(transforms.Resize((h, w))(to_pil(f_1_low)))

        img_var_high = transform_high(img).unsqueeze(0).to(device)
        _, _, _, f_1_high = net(img_var_high)
        f_1_high = f_1_high.detach().squeeze(0).cpu()
        mask_high = np.array(transforms.Resize((h, w))(to_pil(f_1_high)))

    _, thresh_low = cv2.threshold(mask_low, 127, 255, cv2.THRESH_BINARY)
    _, thresh_high = cv2.threshold(mask_high, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(thresh_low, thresh_high)

    if args['crf']:
        mask = crf_refine(np.array(img.convert('RGB')), mask)

    return mask

if __name__ == '__main__':
    mask = infer_mirrornet()
    output_dir = "path_to_output_dir"
    Image.fromarray(mask).save(os.path.join(output_dir, "mask.png"))
    