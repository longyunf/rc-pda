import os
from os.path import join
import argparse
import numpy as np
from functools import reduce

import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

 
def get_intrinsic_matrix(nusc, cam_token):       
    cam_data = nusc.get('sample_data', cam_token)
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    return np.array( cs_rec['camera_intrinsic'] )

    
def current_2_ref_matrix(nusc, current_sd_token, ref_sd_token):
    '''
    inputs:
        current_sd_token: current image token
        ref_sd_token: reference image token      
    '''    
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)    
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])

    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)           
    M_ref_from_global = reduce(np.dot, [ref_from_car, car_from_global])
    
    current_sd_rec = nusc.get('sample_data', current_sd_token)
    current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
    global_from_car = transform_matrix(current_pose_rec['translation'],
                                       Quaternion(current_pose_rec['rotation']), inverse=False)
    
    current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
    car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                        inverse=False)
    
    M_global_from_current = reduce(np.dot, [global_from_car, car_from_current])    
    M_ref_from_current = reduce(np.dot, [M_ref_from_global, M_global_from_current])
    
    return M_ref_from_current

   

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
        
    args = parser.parse_args()

    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '..', 'data')
    dir_nuscenes = os.path.join(args.dir_data, 'nuscenes')


    nusc = NuScenes(args.version, dataroot = dir_nuscenes, verbose=False)        
    dir_data_out = join(args.dir_data, 'prepared_data')
    sample_indices = torch.load(join(args.dir_data,'data_split.tar'))['all_indices'] 
         
    ct = 0         
    for sample_idx in sample_indices:

        cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        
        if cam_data['next']:
                
            cam_token1 = cam_data['next']                    
            cam_data1 = nusc.get('sample_data', cam_token1)            
            cam_token2 = cam_data1['next']        
            
            K = get_intrinsic_matrix(nusc, cam_token1)
            T = current_2_ref_matrix(nusc, cam_token1, cam_token2)
            
        np.savez(join(dir_data_out, '%05d_matrix.npz' % sample_idx), K=K, T=T)    
           
        ct += 1
        print('Save matrix %d/%d' % ( ct, len(sample_indices) ) )
        