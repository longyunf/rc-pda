import argparse
from os.path import join
import numpy as np
import h5py
from tqdm import tqdm
import skimage.io as io

import torch


def create_data_group(hf, mode, sample_indices):
        
    group = hf.create_group('%s' % mode)
        
    im_list = []
    for idx in tqdm(sample_indices, '%s:im' % mode):
        im1 = io.imread(join(dir_label, '%05d_im.jpg' % idx))
        im_list.append(im1)    
    group.create_dataset('im',data=np.array(im_list))
    del im_list
    
    K_list, T_list, msk_lh_list = [],[],[]
    for idx in tqdm(sample_indices, '%s:K,T, msk_lh' % mode):
        matrix = np.load(join(dir_label, '%05d_matrix.npz' % idx))
        K = matrix['K']
        T = matrix['T']
        msk_lh = np.load(join(dir_label, '%05d_msk_lh.npy' % idx))
        
        K_list.append(K)
        T_list.append(T)
        msk_lh_list.append(msk_lh)
    
    group.create_dataset('K',data=np.array(K_list))
    group.create_dataset('T',data=np.array(T_list))  
    group.create_dataset('msk_lh',data=np.array(msk_lh_list))
    del K_list, T_list, msk_lh_list
     
    
    uv2_im_list = []
    for idx in tqdm(sample_indices, '%s:uv2_im' % mode):
        uv2_im = np.load(join(dir_label, '%05d_im_uv.npy' % idx))
        uv2_im_list.append(uv2_im) 
    group.create_dataset('im_uv',data=np.array(uv2_im_list))
    del uv2_im_list
    
    radar_list = []
    for idx in tqdm(sample_indices, '%s:radar_single' % mode):
        radar = np.load(join(dir_label, '%05d_radar_one.npy' % idx) )[...,[0,4,5]].astype('float32')  # d_radar, u2, v2       
        radar_list.append(radar)
    group.create_dataset('radar_one',data=np.array(radar_list))
    del radar_list
    

    radar_list = []
    for idx in tqdm(sample_indices, '%s:radar_short' % mode):
        radar = np.load(join(dir_label, '%05d_radar_short.npy' % idx) )[...,[0,4,5]].astype('float32')  # d_radar, u2, v2       
        radar_list.append(radar)
    group.create_dataset('radar_short',data=np.array(radar_list))
    del radar_list
    
    
    radar_extra_list, msk_moving_list = [],[]
    for idx in tqdm(sample_indices, '%s:radar_extra' % mode):
        radar_extra = np.load(join(dir_label, '%05d_radar_one.npy' % idx) )[...,[1,2]].astype('float32')  # t, rcs 
        msk_moving = np.load(join(dir_label, '%05d_radar_one.npy' % idx) )[...,3].astype('bool')  # moving_msk
        radar_extra_list.append(radar_extra)
        msk_moving_list.append(msk_moving)
        
    group.create_dataset('radar_extra_one',data=np.array(radar_extra_list))
    group.create_dataset('msk_moving_one',data=np.array(msk_moving_list))
    del radar_extra_list, msk_moving_list
    
       
    radar_extra_list, msk_moving_list = [],[]
    for idx in tqdm(sample_indices, '%s:radar_extra_short' % mode):
        radar_extra = np.load(join(dir_label, '%05d_radar_short.npy' % idx) )[...,[1,2]].astype('float32')  # t, rcs 
        msk_moving = np.load(join(dir_label, '%05d_radar_short.npy' % idx) )[...,3].astype('bool')  # moving_msk
        radar_extra_list.append(radar_extra)
        msk_moving_list.append(msk_moving)
 
    group.create_dataset('radar_extra_short',data=np.array(radar_extra_list))
    group.create_dataset('msk_moving_short',data=np.array(msk_moving_list))
    del radar_extra_list, msk_moving_list
    
    gt_list, gt_one_list = [],[]
    for idx in tqdm(sample_indices, '%s:gt' % mode):
        gt = np.load(join(dir_label, '%05d_gt.npy' % idx)).astype('float32')
        gt_list.append(gt)
     
    group.create_dataset('gt',data=np.array(gt_list))   
    group.create_dataset('indices',data=np.array(sample_indices))
    del gt_list
    
    for idx in tqdm(sample_indices, '%s:gt single frame' % mode):
        gt_one = np.load(join(dir_label, '%05d_gt_one.npy' % idx)).astype('float32')
        gt_one_list.append(gt_one)
     
    group.create_dataset('gt_one',data=np.array(gt_one_list)) 
    del gt_one_list
        
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default='/home/longyunf/media/nuscenes', help='dataset directory')
    args = parser.parse_args()
        
    dir_data = args.dir_data
    
    dir_label = join(dir_data, 'prepared_data_dense') 
    path_h5_file = join(dir_data, 'prepared_data_dense.h5')


    train_sample_indices = torch.load(join(dir_data,'data_split_small.tar'))['train_sample_indices']
    val_sample_indices = torch.load(join(dir_data,'data_split_small.tar'))['val_sample_indices']
    test_sample_indices = torch.load(join(dir_data,'data_split_small.tar'))['test_sample_indices']
    
    hf = h5py.File(path_h5_file, 'w')
        
    create_data_group(hf, 'train', train_sample_indices)
    create_data_group(hf, 'val', val_sample_indices)
    create_data_group(hf, 'test', test_sample_indices)
    
    hf.close()
        
        
        

    