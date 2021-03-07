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
        self.radar_raw_list = data['radar_short'][...,0].astype('f4')       
        # self.radar_list = data_radar['radar'][...].astype('f4')
        self.radar_list = data_radar['radar'][...]
        if mode == 'test':
            self.msk_lh_list = data['msk_lh'][...]
                           
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)
        
    def __getitem__(self, idx):
        'Generate one sample of data'
        
        im1 = self.im_list[idx].astype('float32').transpose((2,0,1))       # (3,h,w) 
        d_radar_raw = self.radar_raw_list[idx].astype('float32')[None,...]
        d_radar_multi = self.radar_list[idx].astype('float32')/100             # (5,h,w) # centimeter to meter
     
        d_lidar = self.gt[idx].astype('float32').transpose((2,0,1))        # (1,h,w)
                        
        # d_radar_raw[d_radar_raw>70] = 0               
        # d_radar_multi[d_radar_multi>70] = 0                         
        # # limit gt depth to [0,70]
        # d_lidar[d_lidar>70] = 0 
        
        d_radar_raw[d_radar_raw>50] = 0               
        d_radar_multi[d_radar_multi>50] = 0                         
        # limit gt depth to [0,70]
        d_lidar[d_lidar>50] = 0 
        
        data_in = np.concatenate((im1, d_radar_raw, d_radar_multi), axis=0)    # (9,h,w)
        
 
        if self.mode == 'test':
            msk_lh = self.msk_lh_list[idx].astype('float32')[None, ...]          # (1,h,w)
            sample = {'data_in': data_in, 'd_lidar': d_lidar, 'msk_lh': msk_lh, 'sample_idx': self.indices[idx]}
        else:
            sample = {'data_in': data_in, 'd_lidar': d_lidar}
                           
        return sample


if __name__=='__main__':
    
    
    dir_data= 'd:/Lab/Dataset/nuscenes'   
    path_data_file = join(dir_data, 'prepared_data_dense.h5')
    path_radar_file = join(dir_data, 'enhanced_radar_multi_1_4_2_0.6.h5')
    
    args_train_set = {'path_data_file': path_data_file,
                      'path_radar_file': path_radar_file,
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
   
    
    plt.close('all')
    
    plt.figure()
    plt.imshow(sample['data_in'][0][:3].permute(1,2,0).to(torch.uint8))
    plt.figure()
    plt.imshow(sample['data_in'][0][3])
    
    plt.figure()
    plt.imshow(sample['data_in'][0][4], cmap='jet')
    plt.colorbar()    
    plt.figure()
    plt.imshow(sample['data_in'][0][5], cmap='jet')
    plt.colorbar()    
    plt.figure()
    plt.imshow(sample['data_in'][0][6], cmap='jet')
    plt.colorbar()    
    plt.figure()
    plt.imshow(sample['data_in'][0][7], cmap='jet')
    plt.colorbar()    
    plt.figure()
    plt.imshow(sample['data_in'][0][8], cmap='jet')
    plt.colorbar()

    

    

    
    




