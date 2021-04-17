import torch
from torch.utils import data
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import h5py
import os

      
def cal_uv1(h, w, K, downsample_scale=4, y_cutoff=33):
    '''
    uv_map: h x w x 2
    '''
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map, y_map = x_map.astype('float32'), y_map.astype('float32')
    
    cx = cx / downsample_scale
    cy = cy / downsample_scale - y_cutoff
    f = f / downsample_scale
    
    u_map = (x_map - cx) / f
    v_map = (y_map - cy) / f
        
    uv_map = np.stack([u_map,v_map], axis=2)
    
    return uv_map


def cal_uv_translation(uv2, R, msk_uv2=None):
    '''
    inputs:
        uv2: (2 x h x w); full flow
        R: rotaion matrix (from u1,v1 -> u2,v2)
        msk_uv2: h x w
    output:
        uvt2: flow from translation inv(R)*t
    
    '''
    
    u2, v2 = uv2[0], uv2[1]

    R_inv = np.linalg.inv(R)
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = R_inv.flatten()
    ut = (u2*r11 + v2*r12 + r13) / (u2*r31 + v2*r32 +r33)
    vt = (u2*r21 + v2*r22 + r23) / (u2*r31 + v2*r32 +r33)
    
    if msk_uv2 is not None:
        ut = ut * msk_uv2
        vt = vt * msk_uv2
        
    uvt2 = np.stack([ut,vt])

    return uvt2



def init_data_loader(args, mode):
    
    if mode == 'train':
        batch_size = args.batch_size
        if args.no_data_shuffle:
            shuffle = False
        else:
            shuffle = True
    else:
        batch_size = args.test_batch_size
        shuffle = False
        
    args_dataset = {'path_data_file': args.path_data_file,
                    'mode': mode}
    args_data_loader = {'batch_size': batch_size,
                       'shuffle': shuffle,
                       'num_workers': args.num_workers}
    dataset = Dataset(**args_dataset)    
    data_loader = torch.utils.data.DataLoader(dataset, **args_data_loader)
    
    return data_loader
    

class Dataset(data.Dataset):     
    def __init__(self, path_data_file, mode):               
        'Initialization'   
        
        self.mode = mode
        data = h5py.File(path_data_file, 'r')[mode] 
        self.im_list = data['im'][...]
        self.K_list = data['K']
        self.T_list = data['T']
        self.uv2_im_list = data['im_uv'][...].astype('f4')
        self.radar_list = data['radar'][...].astype('f4')
        self.gt = data['gt'][...,[0]].astype('f4')
        self.indices = data['indices']
        if mode == 'test':
            self.msk_lh_list = data['msk_lh']
                           
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)
        
    def __getitem__(self, idx):
        'Generate one sample of data'
        
        im1 = self.im_list[idx].astype('float32').transpose((2,0,1))/255   # (3,h,w)
        h, w = im1.shape[1:]    
        
        K = self.K_list[idx]
        R = self.T_list[idx][:3,:3]
        
        uv1_map = cal_uv1(h, w, K, downsample_scale=4, y_cutoff=33).transpose((2,0,1))  # (2,h,w)             
        uv2_im = self.uv2_im_list[idx].astype('float32').transpose((2,0,1))  # (2,h,w)        
        
        d_radar = self.radar_list[idx][...,[0]].astype('float32').transpose((2,0,1))      # (1,h,w)
        uv2_radar = self.radar_list[idx][...,[1,2]].astype('float32').transpose((2,0,1))  # (2,h,w)      
        
        d_lidar = self.gt[idx].astype('float32').transpose((2,0,1))
                

        # limit gt depth to [0,50]
        d_lidar[d_lidar>50] = 0 
        # filter radar detph
        d_radar[d_radar>50] = 0     
        msk_radar = d_radar[0] > 0              # h x w
        d_radar_norm = d_radar/50               # normalized to 0-1
        
        scale_factor = 30
        
        ## remove rotational flow
        uvt2_im = cal_uv_translation(uv2_im, R)
        uvt2_radar = cal_uv_translation(uv2_radar, R, msk_radar)
        
        duv_im = (uvt2_im - uv1_map) * scale_factor                    # optical flow
        duv_radar = (uvt2_radar - uv1_map) * msk_radar * scale_factor  # radar flow
        
        data_in = np.concatenate((im1, d_radar_norm, uv1_map, duv_im, duv_radar), axis=0)    # (10,h,w)
        
        if self.mode == 'test':
            msk_lh = self.msk_lh_list[idx].astype('float32')[None, ...]          # (1,h,w)
            sample = {'data_in': data_in, 'd_lidar': d_lidar, 'd_radar': d_radar, 'msk_lh': msk_lh}
        else:
            sample = {'data_in': data_in, 'd_lidar': d_lidar, 'd_radar': d_radar}
                           
        return sample


if __name__=='__main__':
    
    this_dir = os.path.dirname(__file__)
    dir_data = join(this_dir, '..', 'data')  
    path_data_file = join(dir_data, 'prepared_data.h5')
    args_train_set = {'path_data_file': path_data_file,
                      'mode': 'train'}
    args_train_loader = {'batch_size': 6,
                         'shuffle': True,
                         'num_workers': 0}
  
    train_set = Dataset(**args_train_set)    
    train_loader = torch.utils.data.DataLoader(train_set, **args_train_loader)
    
    data_iterator = enumerate(train_loader)
    
    batch_idx, sample = next(data_iterator)
    
    print('batch_idx', batch_idx)
    print('data_in', sample['data_in'].shape, type(sample['data_in']),sample['data_in'].dtype)
    print('d_lidar', sample['d_lidar'].shape, type(sample['d_lidar']), sample['d_lidar'].dtype)
    print('d_lidar', sample['d_radar'].shape, type(sample['d_radar']), sample['d_radar'].dtype)

    

    