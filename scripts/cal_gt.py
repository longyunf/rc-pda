'''
   Generate ground truth depth.

'''

import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from timeit import default_timer as timer

from nuscenes.nuscenes import NuScenes

import _init_paths
from fuse_lidar import merge_lidar, get_tracks_forward, update_key_tracks, update_non_key_tracks, cal_depthMap_flow, filter_occlusion, lidarFlow2uv, filter_occlusion_by_bbox


def get_tracks_scene(nusc, sample_idx):
    '''
    obtain all bounding boxes of all Lidar frames in a scenes
    
    '''    
    box_tracks_key = {}
    box_tracks_non = {}
    
    idx_temp = sample_idx        
    if nusc.sample[idx_temp]['next'] != '':    
        track_forward_key, track_forward_non = get_tracks_forward(nusc, idx_temp)  
        box_tracks_key = update_key_tracks(box_tracks_key, track_forward_key)
        box_tracks_non = update_non_key_tracks(box_tracks_non, track_forward_non)
        while True:
            idx_temp += 1
            if nusc.sample[idx_temp]['next'] == '':                
                break
            track_forward_key, track_forward_non = get_tracks_forward(nusc, idx_temp)
            box_tracks_key = update_key_tracks(box_tracks_key, track_forward_key)
            box_tracks_non = update_non_key_tracks(box_tracks_non, track_forward_non)
               
    idx_temp = sample_idx        
    if nusc.sample[idx_temp]['prev'] != '':    
        while True:
            idx_temp -= 1
            track_forward_key, track_forward_non = get_tracks_forward(nusc, idx_temp)
            box_tracks_key = update_key_tracks(box_tracks_key, track_forward_key)
            box_tracks_non = update_non_key_tracks(box_tracks_non, track_forward_non)
            if nusc.sample[idx_temp]['prev'] == '':
                break
                 
    box_tracks_key.update(box_tracks_non)            
    box_tracks = box_tracks_key
    
    return box_tracks
  

if __name__ == '__main__':
    np.random.seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    
    args = parser.parse_args()    

    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '..', 'data')
    dir_nuscenes = os.path.join(args.dir_data, 'nuscenes')
    start_idx = args.start_idx
    end_idx = args.end_idx

    
    nusc = NuScenes(args.version, dataroot = dir_nuscenes, verbose=False)
    dir_data_out = join(args.dir_data, 'prepared_data')
    sample_indices = torch.load(join(args.dir_data,'data_split.tar'))['all_indices']
   
    downsample_scale = 4
    y_cutoff = 33
       
    N_total = len(sample_indices)
    print('Total sample number:', N_total)
    if start_idx == None:
        start_idx = 0
    
    if end_idx == None:
        end_idx = N_total - 1
    
    if end_idx > N_total - 1 :
        end_idx = N_total - 1
    
    start_sample_idx = sample_indices[start_idx]
    box_tracks = get_tracks_scene(nusc, start_sample_idx)
    current_scene_token = nusc.sample[start_sample_idx]['scene_token']
        
    ct = 0
    for sample_idx in sample_indices[start_idx: end_idx+1]:
        
        start = timer()
        
        flow_im = np.load(join(dir_data_out, '%05d_flow.npy' % sample_idx))
        matrix = np.load(join(dir_data_out, '%05d_matrix.npz' % sample_idx))
        seg = np.load(join(dir_data_out, '%05d_seg.npy' % sample_idx))
        
        K = matrix['K']
                       
        if nusc.sample[sample_idx]['scene_token'] != current_scene_token:
            print('New scene')
            current_scene_token = nusc.sample[sample_idx]['scene_token']
            box_tracks = get_tracks_scene(nusc, sample_idx)
        
                
        n_forward = 42
        n_skip = 1
        if nusc.sample[sample_idx]['next'] == '' or nusc.sample[sample_idx + 1]['next'] == '' or nusc.sample[sample_idx + 2]['next'] == '':            
            n_backward = 21            
        else:
            n_backward = 9
                         
        x1, y1, depth1, x2, y2, depth2, msk_low_h, msk_in, x_cn, y_cn, depth_cn, msk_cn = merge_lidar(nusc, sample_idx, n_backward, n_forward, box_tracks, n_skip)
        
        depth_map, flow_lidar, msk_map_low_h, msk_map_in = cal_depthMap_flow(x1, y1, depth1, x2, y2, depth2, msk_low_h, msk_in, downsample_scale=4, y_cutoff=33)
                
        depth_map, msk_d1 = filter_occlusion_by_bbox(depth_map, seg, msk_map_in, x_cn, y_cn, depth_cn, msk_cn, downsample_scale=4, y_cutoff=33)
        flow_lidar, msk_map_low_h = flow_lidar * msk_d1[..., None], msk_map_low_h * msk_d1

        depth_map, msk_map_low_h, msk_map_in, flow_lidar = filter_occlusion(depth_map, msk_map_low_h, msk_map_in, flow_lidar, flow_im, thres = 3)

        uv2 = lidarFlow2uv(flow_lidar, K, depth_map, downsample_scale=4, y_cutoff=33)
                
        depth_map = depth_map[..., None]       
        gt = np.concatenate((depth_map, uv2), axis=2)        
      
        np.save(join(dir_data_out, '%05d_gt.npy' % sample_idx), gt)        
        np.save(join(dir_data_out, '%05d_msk_lh.npy' % sample_idx), msk_map_low_h) 
                
        ct += 1
        print('compute depth %d/%d' % ( ct, end_idx - start_idx + 1 ) )
        
        end = timer()
        t = end-start     
        print('Time used: %.1f s' % t)
        
    