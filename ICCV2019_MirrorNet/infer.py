"""
 @Time    : 2026 Update (Original: 9/29/19)
 @Author  : TaylorMei (Updated for 2-Scale Pyramid Inference & CPU optimization)
 @Project : ICCV2019_MirrorNet
 @File    : infer_pyramid.py
"""
import numpy as np
import os
import time
import cv2

import torch
from PIL import Image
from torchvision import transforms

from config import msd_testing_root
from misc import check_mkdir, crf_refine
from mirrornet import MirrorNet

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

to_test = {'MSD': msd_testing_root}
to_pil = transforms.ToPILImage()

def main():
    net = MirrorNet().to(device)

    if len(args['snapshot']) > 0:
        snapshot_file = os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')
        print('Load snapshot {} for testing'.format(args['snapshot']))
        net.load_state_dict(torch.load(snapshot_file, map_location=device))
        print('Load {} succeed!'.format(snapshot_file))

    net.eval()
    
    transform_low = get_transform(args['scale_low'])
    transform_high = get_transform(args['scale_high'])

    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image'))]
            start = time.time()
            
            for idx, img_name in enumerate(img_list):
                print('Predicting 2-scale pyramid for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                output_dir = os.path.join(root, 'result')
                check_mkdir(output_dir)
                
                img = Image.open(os.path.join(root, 'image', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                w, h = img.size

                # =============================================================
                # SCALA 1: Inferenza Ridotta (384x384)
                # =============================================================
                img_var_low = transform_low(img).unsqueeze(0).to(device)
                _, _, _, f_1_low = net(img_var_low)
                f_1_low = f_1_low.detach().squeeze(0).cpu()
                mask_low = np.array(transforms.Resize((h, w))(to_pil(f_1_low)))

                # =============================================================
                # SCALA 2: Inferenza Maggiorata (640x640) 
                # =============================================================
                img_var_high = transform_high(img).unsqueeze(0).to(device)
                _, _, _, f_1_high = net(img_var_high)
                f_1_high = f_1_high.detach().squeeze(0).cpu()
                mask_high = np.array(transforms.Resize((h, w))(to_pil(f_1_high)))

                _, thresh_low = cv2.threshold(mask_low, 127, 255, cv2.THRESH_BINARY)
                _, thresh_high = cv2.threshold(mask_high, 127, 255, cv2.THRESH_BINARY)
                mask_piramidale = cv2.bitwise_or(thresh_low, thresh_high)

                if args['crf']:
                    mask_piramidale = crf_refine(np.array(img.convert('RGB')), mask_piramidale)

                Image.fromarray(mask_piramidale).save(os.path.join(output_dir, img_name[:-4] + ".png"))

            end = time.time()
            print("Average Time per 2-Scale Inference: {:.2f}s".format((end - start) / len(img_list)))

if __name__ == '__main__':
    main()