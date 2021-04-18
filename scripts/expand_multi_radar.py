import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from os.path import join
import sys
from pyramidNet import PyramidCNN
from tqdm import tqdm
import h5py

from data_loader_aff import init_data_loader
from aff_utils import neighbor_connection
from test_aff import cal_enhanced_depth_with_max_aff



def load_weights(args, model):
    f_checkpoint = join(args.dir_result, 'checkpoint.tar')        
    if os.path.isfile(f_checkpoint):
        print('load best model')        
        model.load_state_dict(torch.load(f_checkpoint)['state_dict'])
        # model.load_state_dict(torch.load(f_checkpoint)['state_dict_best'])
    else:
        sys.exit('No model found')
 
    
def init_env():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    cudnn.benchmark = True if use_cuda else False
    return device


        

def create_data_group(hf, mode, loader, device, nb, model, thres_affs):
    
           
    group = hf.create_group('%s' % mode)        
    radar_list = []
    with torch.no_grad():
        for ct, sample in enumerate( tqdm(loader, '%s' % mode) ):
            data_in, d_radar = sample['data_in'].to(device), sample['d_radar'].to(device)                              
            prd = torch.sigmoid( model(data_in)[0] ) 
            
            d_est, aff = cal_enhanced_depth_with_max_aff(prd, d_radar, nb, device, thres_affs[0])
            
            d_est_multi = []
            
            for thres in thres_affs:
                msk = aff > thres  
                
                d = (d_est * msk * 100).round().astype(np.int16)
                d_est_multi.append(d)
                
            data = np.array(d_est_multi)
            # print(data.shape, data.dtype)
                
            radar_list.append(data)
    
    group.create_dataset('radar',data=np.array(radar_list))
    del radar_list   
 
    
 
    
def main(args):
    # args.dir_data = 'd:/Lab/Dataset/nuscenes'
    # # args.dir_result = join(args.dir_data, 'train_result', 'aff')
    # args.nChannels = 32
    # args.nLevels = 4
    # args.left_right, args.top, args.bottom = 1, 4, 2
    
    # args.dir_data = '/mnt/home/longyunf/scratch18/nuscenes'
    # args.dir_result = '/mnt/home/longyunf/train_results/aff_3_24_8'
    # args.left_right, args.top, args.bottom = 3, 24, 8
    
    # args.left_right, args.top, args.bottom = 3, 20, 5
    args.left_right, args.top, args.bottom = 1, 8, 1
    
    
    
    args.outChannels = ( args.left_right * 2 + 1 ) * (args.top + args.bottom + 1)
    print('output channels: ', args.outChannels)  
    
    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result', 'aff_%d_%d_%d' % (args.left_right, args.top, args.bottom))
    args.path_data_file = join(args.dir_data, 'prepared_data_dense.h5') 
    
                
    device = init_env()
    
       
    model = PyramidCNN(args.nLevels, args.nPred, args.nPerBlock, 
                        args.nChannels, args.inChannels, args.outChannels, 
                        args.doRes, args.doBN, doELU=False, 
                        predPix=False, predBoxes=False).to(device)
    
    load_weights(args, model)       
    model.eval()
    
    thres_affs = [0.6, 0.7, 0.8, 0.9, 0.95]  
    # thres_affs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]  # multi2    
    # thres_affs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # multi3
    nb = neighbor_connection(*(args.left_right, args.left_right, args.top, args.bottom))
    
    
    path_h5_file = join(args.dir_data, 'enhanced_radar_multi_%d_%d_%d_%.1f.h5' % (args.left_right, args.top, args.bottom, thres_affs[0]) )
    # path_h5_file = join(args.dir_data, 'enhanced_radar_multi2_%d_%d_%d_%.1f.h5' % (args.left_right, args.top, args.bottom, thres_affs[0]) )
    # path_h5_file = join(args.dir_data, 'enhanced_radar_multi3_%d_%d_%d_%.1f.h5' % (args.left_right, args.top, args.bottom, thres_affs[0]) )
    hf = h5py.File(path_h5_file, 'w')
    
    
    train_loader = init_data_loader(args, 'train')
    create_data_group(hf, 'train', train_loader, device, nb, model, thres_affs)
    del train_loader

    val_loader = init_data_loader(args, 'val')
    create_data_group(hf, 'val', val_loader, device, nb, model, thres_affs)
    del val_loader
    
    test_loader = init_data_loader(args, 'test')
    create_data_group(hf, 'test', test_loader, device, nb, model, thres_affs)
    del test_loader
    
    hf.close()
    
       
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')    
    parser.add_argument('--dir_data', type=str, default='/home/longyunf/media/nuscenes', help='prepared data directory')
    parser.add_argument('--dir_result', type=str, help='directory for training results')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_data_shuffle', type=bool, default=True, help='for generating training data in order')
    
    parser.add_argument('--nLevels', type=int, default=5)
    parser.add_argument('--nPred', type=int, default=1)
    parser.add_argument('--nPerBlock', type=int, default=2)
    parser.add_argument('--nChannels', type=int, default=80)   
    parser.add_argument('--inChannels', type=int, default=10)
    # parser.add_argument('--outChannels', type=int, default=55, help='number of output channel of network; automatically set to 1 if pred_task is foreground_seg')
    parser.add_argument('--doRes', type=bool, default=True)
    parser.add_argument('--doBN', type=bool, default=True)    
    
     # neighborhood
    parser.add_argument('--left_right', type=int, default=2)
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--bottom', type=int, default=5)
    
    args = parser.parse_args()
    
    main(args)
    
   
