'''
radar label: h x w x 6
depth_map, time_map, rcs_map, moving_msk, u2, v2

merge radar frames in 0.3s

'''

from nuscenes.nuscenes import NuScenes
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import argparse
from timeit import default_timer as timer
import torch

from fuse_radar4b import merge_selected_radar, cal_depthMap_flow, radarFlow2uv


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default='d:/Lab/Dataset/nuscenes', help='dataset directory')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='dataset split')
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    
    args = parser.parse_args()    
    dir_data = args.dir_data    
    version = args.version
    start_idx = args.start_idx
    end_idx = args.end_idx
    
       
    downsample_scale = 4
    y_cutoff = 33
   
    nusc = NuScenes(version, dataroot = dir_data, verbose=False)
    dir_data_out = join(dir_data, 'prepared_data_dense')   
        
    sample_indices = torch.load(join(dir_data,'data_split_small.tar'))['all_indices']
       
    N_total = len(sample_indices)
    print('Total sample number:', N_total)
        
    if start_idx == None:
        start_idx = 0
        
    if end_idx == None or end_idx > N_total - 1:
        end_idx = N_total -1
    
    frm_range = [0,4]
    
    ct = 0
    for sample_idx in sample_indices[start_idx: end_idx+1]:
    # for sample_idx in sample_indices[100:102]:
        
        start = timer()
        matrix = np.load(join(dir_data_out, '%05d_matrix.npz' % sample_idx))
        K = matrix['K']
        
        x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp= merge_selected_radar(nusc, sample_idx, frm_range)
                
        depth_map1, flow, time_map1, rcs_map1, v_comp_map1 = cal_depthMap_flow(x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp, downsample_scale=4, y_cutoff=33)        
        uv2 = radarFlow2uv(flow, K, depth_map1, downsample_scale=4, y_cutoff=33)
        
        
        depth_map = np.stack([depth_map1, time_map1, rcs_map1, v_comp_map1], axis = 2)     
        radar_data = np.concatenate((depth_map, uv2), axis=2) 
        
        np.save(join(dir_data_out, '%05d_radar_short.npy' % sample_idx), radar_data) 
         
        ct += 1
        print('compute depth %d/%d' % ( ct, end_idx - start_idx + 1 ) )
        
        end = timer()
        t = end-start     
        print('Time used: %.1f s' % t)
        
        