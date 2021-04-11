import argparse
import os
import glob
import numpy as np
import torch
from os.path import join
from skimage.transform import resize

def flow2uv(flow, K, downsample_scale=4, y_cutoff=33):
    '''
    uv_map: h x w x 2
    '''
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    
    h,w = flow.shape[:2]
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map, y_map = x_map.astype('float32'), y_map.astype('float32')
    x_map += flow[..., 0]
    y_map += flow[..., 1]
    
    cx = cx / downsample_scale
    cy = cy / downsample_scale - y_cutoff
    f = f / downsample_scale
    
    u_map = (x_map - cx) / f
    v_map = (y_map - cy) / f
    
    uv_map = np.stack([u_map,v_map], axis=2)
    
    return uv_map

   

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)   
    args = parser.parse_args()

    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '..', 'data')
     
        
    out_dir = join(args.dir_data, 'prepared_data')
    flow_list = np.array(np.sort(glob.glob(join(out_dir, '*flow.npy'))))
    
    N = len(flow_list)
    
    downsample_scale = 4
    y_cutoff = 33
    
    ct = 0    
    for f_flow in flow_list:
        flow = np.load(f_flow)
        
        matrix = np.load(f_flow[:-8] + 'matrix.npz')
        K = matrix['K']
        
        uv_map = flow2uv(flow, K, downsample_scale, y_cutoff)
        
        np.save(f_flow[:-8] + 'im_uv.npy', uv_map)
        
        ct += 1
        print('Processing %d/%d' % ( ct, N ) )
        
