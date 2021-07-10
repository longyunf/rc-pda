import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from os.path import join
import sys
from tqdm import tqdm

import _init_paths
from data_loader_depth_hg import init_data_loader
from hourglassNet import network



def load_weights(args, model):
    f_checkpoint = join(args.dir_result, 'checkpoint.tar')        
    if os.path.isfile(f_checkpoint):
        print('load best model')        
        model.load_state_dict(torch.load(f_checkpoint)['state_dict_best'])
    else:
        sys.exit('No model found')
 
    
def init_env():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    cudnn.benchmark = True if use_cuda else False
    return device


def prd_one_sample(model, test_loader, device, idx, args):
        
    with torch.no_grad():
        for ct, sample in enumerate(test_loader):
            if ct == idx:
                im, d_radar, d_lidar = sample['im'].to(device), sample['d_radar'].to(device), sample['d_lidar'].to(device)            
                prd = model(d_radar, im)[0]
                
                im = im[0].permute(1,2,0).to('cpu').numpy().astype('uint8')
                d_radar = d_radar[0][0].to('cpu').numpy()
                d_lidar = d_lidar[0][0].to('cpu').numpy()               
                prd = prd[0][0].cpu().numpy()
                
                break
            
            
    d_max = args.d_max
    d_min = 1e-3
    
    
    prd = prd.clip(d_min, d_max)    
    msk_valid = np.logical_and(d_lidar>d_min, d_lidar<d_max)    
    d_lidar = d_lidar * msk_valid    
    
    d_error = np.abs( d_lidar - prd ) * msk_valid   
    
    depth = np.concatenate([d_lidar, prd], axis = 0)
      
    plt.close('all')   
    
    plt.figure()
    plt.imshow(im)
    plt.show()
    
    plt.figure()
    plt.imshow(d_radar, cmap='jet')
    plt.title('radar')
    plt.colorbar()
    plt.show()
   
    plt.figure()
    plt.imshow(depth, cmap='jet')
    plt.title('depth')
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.imshow(prd, cmap='jet')
    plt.title('Final depth')
    plt.colorbar()
    plt.axis('off')
    plt.show()
        
    plt.figure()
    plt.imshow(d_error, cmap='jet')
    plt.title('error')
    plt.colorbar()
    plt.show()



def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    
    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)
    
    mae = np.mean(np.abs(gt - pred))
       
    return silog, log10, mae, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3



def evaluate(model, test_loader, device, d_min=1e-3, d_max=70, eval_low_height = False):
    
    model.eval()
    errors = np.zeros(10)   
    
    with torch.no_grad():
        for sample in tqdm(test_loader, 'Evaluation'):
            im, d_radar, gt = sample['im'].to(device), sample['d_radar'].to(device), sample['d_lidar']            
            prd = ( torch.clamp( model(d_radar, im)[0], min=d_min, max=d_max ) ).to('cpu').numpy()
            
            if eval_low_height:
                gt = gt * sample['msk_lh']
                
            gt = gt.numpy()             
            msk_valid = np.logical_and(gt>d_min, gt<d_max)
            
            errors += compute_errors(gt[msk_valid], prd[msk_valid])           
     
    errors /= len(test_loader) 
       
    print(' \n silog: %f, log10: %f, mae: %f, abs_rel: %f, sq_rel: %f, rmse: %f, rmse_log: %f, d1: %f, d2: %f, d3: %f' \
          % (errors[0], errors[1], errors[2], errors[3], errors[4], errors[5], errors[6], errors[7], errors[8], errors[9]))
   
          
def main(args):
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
  
    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result', 'depth_completion_hourglass')
    args.path_data_file = join(args.dir_data, 'prepared_data.h5') 
    
    args.path_radar_file = join(args.dir_data, 'mer_2_30_5_0.5.h5')
                
    device = init_env()
        
    test_loader = init_data_loader(args, 'test')

    model = network(rd_layers = 7).to(device)

    load_weights(args, model)       
    model.eval()
            
    idx = 150
    prd_one_sample(model, test_loader, device, idx, args)
        
    # evaluation
    evaluate(model, test_loader, device, d_min=1e-3, d_max=args.d_max, eval_low_height = False)

    print('\n Low height')
    evaluate(model, test_loader, device, d_min=1e-3, d_max=args.d_max, eval_low_height = True)
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_result', type=str)

    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    
    parser.add_argument('--d_max', type=float, default=50)
    args = parser.parse_args()
    
    main(args)
    
   