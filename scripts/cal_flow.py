'''
Compute flow
Based on RAFT (https://github.com/princeton-vl/RAFT)

'''

import os
import sys
import argparse
import glob
import numpy as np
from os.path import join

import torch
from PIL import Image

raft_path = join(os.path.dirname(__file__), '..', 'external', 'RAFT', 'core')
if raft_path not in sys.path:
    sys.path.insert(0, raft_path)
from raft import RAFT

DEVICE = 'cuda'

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')    
    parser.add_argument('--dir_data', type=str, help='dataset directory')
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
        
    args = parser.parse_args()
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
        
    if args.model == None:
        this_dir = os.path.dirname(__file__)
        args.model = join(this_dir, '..', 'external', 'RAFT', 'models', 'raft-kitti.pth')
    
    start_idx = args.start_idx
    end_idx = args.end_idx
        
    out_dir = join(args.dir_data, 'prepared_data')
       
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
        
    im_list = np.array(np.sort(glob.glob(join(out_dir, '*im.jpg'))))
    
    N = len(im_list)
        
    print('Total sample number:', N)
    
    if start_idx == None:
        start_idx = 0
    
    if end_idx == None or end_idx > N - 1 :
        end_idx = N - 1
        
    ct = 0
    for sample_idx in range(start_idx, end_idx + 1):
        
        f_im1 = im_list[sample_idx]
        
        im1 = np.array(Image.open(f_im1)).astype(np.uint8)
        
        f_im_next = f_im1[:-4] + '_next.jpg'

        im2 = np.array(Image.open(f_im_next)).astype(np.uint8) 
                       
        im1 = torch.from_numpy(im1).permute(2, 0, 1).float()
        im1 = im1[None,].to(DEVICE)   
        
        im2 = torch.from_numpy(im2).permute(2, 0, 1).float()
        im2 = im2[None,].to(DEVICE)   
    
        with torch.no_grad():
            flow_low, flow_up = model(im1, im2, iters=20, test_mode=True)
            flow = flow_up[0].permute(1,2,0).cpu().numpy()
            
            path_flow = f_im1[:-6] + 'flow.npy'
            
            np.save(path_flow, flow)
        
        ct += 1
        print('compute flow %d/%d' % ( ct, N ) )
    
