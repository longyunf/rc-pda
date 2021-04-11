'''
This is based on RAFT (code and models are available from https://github.com/princeton-vl/RAFT)

compute flow from im1 to im2

'''

import sys
sys.path.append('/home/longyunf/Repos/RAFT/core')


import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from os.path import join


DEVICE = 'cuda'

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/longyunf/Repos/RAFT/models/raft-kitti.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')    
    parser.add_argument('--dir_data', type=str, default='/home/longyunf/media/nuscenes', help='dataset directory')
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    
    
    args = parser.parse_args()
    
    dir_data = args.dir_data   
    start_idx = args.start_idx
    end_idx = args.end_idx
    
    
    out_dir = join(dir_data, 'prepared_data_dense')
    
    
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
        f_im_prev = f_im1[:-4] + '_prev.jpg'

        if os.path.exists(f_im_next):
            im2 = np.array(Image.open(f_im_next)).astype(np.uint8) 
        else:
            im2 = np.array(Image.open(f_im_prev)).astype(np.uint8)
                
        # # Pads images such that dimensions are divisible by 8
        # im1 = np.pad(im1, ((2,2),(0,0),(0,0)), 'constant')
        # im2 = np.pad(im2, ((2,2),(0,0),(0,0)), 'constant')
                
        im1 = torch.from_numpy(im1).permute(2, 0, 1).float()
        im1 = im1[None,].to(DEVICE)   
        
        im2 = torch.from_numpy(im2).permute(2, 0, 1).float()
        im2 = im2[None,].to(DEVICE)   
    
        with torch.no_grad():
            flow_low, flow_up = model(im1, im2, iters=20, test_mode=True)
            # flow_low, flow_up = model(im2, im1, iters=20, test_mode=True)
            # flow = flow_up[0].permute(1,2,0).cpu().numpy()[2:-2,...]
            flow = flow_up[0].permute(1,2,0).cpu().numpy()
            
            path_flow = f_im1[:-6] + 'flow.npy'
            
            np.save(path_flow, flow)
        
        ct += 1
        print('compute flow %d/%d' % ( ct, N ) )
    
