import torch
from torch.utils import data
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import h5py
   
def init_data_loader(args, mode):
    
    if mode == 'train':
        batch_size = args.batch_size
        shuffle = True
    else:
        batch_size = args.test_batch_size
        shuffle = False
        
    args_dataset = {'path_data_file': args.path_data_file,
                    'path_radar_file': args.path_radar_file,
                    'mode': mode}
    args_data_loader = {'batch_size': batch_size,
                       'shuffle': shuffle,
                       'num_workers': args.num_workers}
    dataset = Dataset(**args_dataset)    
    data_loader = torch.utils.data.DataLoader(dataset, **args_data_loader)
    
    return data_loader
    

class Dataset(data.Dataset):     
    def __init__(self, path_data_file, path_radar_file, mode):               
        'Initialization'   
        
        self.mode = mode
        data = h5py.File(path_data_file, 'r')[mode] 
        data_radar = h5py.File(path_radar_file, 'r')[mode] 
        
        self.im_list = data['im'][...]        
        self.gt = data['gt'][...,[0]].astype('f4')
        self.indices = data['indices']        
        self.radar_raw_list = data['radar'][...,0].astype('f4')       
        self.radar_list = data_radar['radar'][...]        
        if mode == 'test':
            self.msk_lh_list = data['msk_lh'][...]
                           
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)
        
    def __getitem__(self, idx):
        'Generate one sample of data'
        
        im = self.im_list[idx].astype('float32').transpose((2,0,1))
        d_radar_raw = self.radar_raw_list[idx].astype('float32')[None,...]
        d_radar_multi = self.radar_list[idx].astype('float32')/100         # centimeter to meter
     
        d_lidar = self.gt[idx].astype('float32').transpose((2,0,1))        # (1,h,w)
        
        d_radar_raw[d_radar_raw>50] = 0               
        d_radar_multi[d_radar_multi>50] = 0                         
        d_lidar[d_lidar>50] = 0 
        
        d_radar = np.concatenate((d_radar_raw, d_radar_multi), axis=0)
        
        if self.mode == 'test':
            msk_lh = self.msk_lh_list[idx].astype('float32')[None, ...]
            sample = {'im': im, 'd_radar': d_radar, 'd_lidar': d_lidar, 'msk_lh': msk_lh, 'sample_idx': self.indices[idx]}
        else:
            sample = {'im': im, 'd_radar': d_radar, 'd_lidar': d_lidar}
                           
        return sample
