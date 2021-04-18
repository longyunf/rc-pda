
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from os.path import join
import sys
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import copy

import torch
import torch.backends.cudnn as cudnn

import _init_paths
from data_loader_pda import init_data_loader
from pda_utils import depth_to_connect, neighbor_connection, otherHalf, cal_nb_depth
from train_pda import BCE_loss
from pyramidNet import PyramidCNN



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



def plt_depth_on_im(depth_map, im, title = '',  ptsSize = 1):
    
    h,w = im.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    msk = depth_map > 0
    
    plt.figure()
    plt.imshow(im) 
    plt.scatter(x_map[msk], y_map[msk], c=depth_map[msk], s=ptsSize, cmap='jet')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')



def cal_enhanced_depth_with_max_aff(prd_aff, d_radar, nb, device, thres_aff=0.5):
    '''
    input:
        prd_aff: tensor: (n_batch, n_nb, h, w) 
        d_radar: tensor: (n_batch, 1, h, w)
    output:
        d_est: numpy (h,w)
        aff_max: numpy (h,w); the maximum affinity associated with the depth
    
    '''    
    nb_aff = otherHalf(prd_aff, nb.xy)
    nb2 = copy.deepcopy(nb)
    nb2.reflect()

    nb_depth = cal_nb_depth(d_radar, nb2, device)
    
    nb_aff[nb_aff <= thres_aff] = 0
    nb_aff[nb_depth == 0] = 0
    
    
    aff_max, _ = torch.max(nb_aff, dim=1, keepdim=True)    
    msk_max = ( aff_max.eq( nb_aff ) ) & (aff_max>0)
    
    n_max = torch.sum(msk_max, dim=1)
    n_max[n_max==0] = -1
    
    d_est = torch.sum( nb_depth * msk_max, dim=1) / n_max  
    
    
    d_est = d_est.squeeze().cpu().numpy()
    aff_max = aff_max.squeeze().cpu().numpy()
    
    return d_est, aff_max

    
def cal_precision_recall(prd, connected, thres_score = 0.5):
    
    
    msk_prd = prd > thres_score
    msk_connected = connected == 1
    
    tp = np.sum(msk_prd * msk_connected)       
    det_p = np.sum(msk_prd)               
    all_p = np.sum(msk_connected)         
    
    return tp, det_p, all_p
    

def evaluate(model, nb, test_loader, device):
    ave_loss = 0 
    ave_ap = 0
    sum_tp = sum_det_p = sum_all_p = 0
      
    thres_score = 0.5
    
    with torch.no_grad():
        for sample in tqdm(test_loader, 'Evaluation'):
            data_in, d_lidar, d_radar = sample['data_in'].to(device), sample['d_lidar'].to(device), sample['d_radar'].to(device)               
            connected = depth_to_connect(d_radar, d_lidar, nb, device)                
            prd = model(data_in)[0]           
            msk = connected >= 0
            
            ave_loss += BCE_loss(prd, connected).item()
            prd = torch.sigmoid( prd ) 
            ave_ap += average_precision_score(connected[msk].cpu().numpy(), prd[msk].cpu().numpy())
            tp, det_p, all_p = cal_precision_recall(prd[msk].cpu().numpy(), connected[msk].cpu().numpy(), thres_score)
            sum_tp += tp
            sum_det_p += det_p
            sum_all_p += all_p
            
    ave_ap /= len(test_loader)  
    ave_loss /= len(test_loader)
    
    precision = sum_tp / sum_det_p
    recall = sum_tp / sum_all_p
    

    print('\n ave_ap: %.4f; ave_loss: %f' % (ave_ap, ave_loss))  
    print('\n precision: %.2f; recall: %.2f' % (precision, recall))  
    
    
    
def main(args):
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
         
    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result', 'pda_%d_%d_%d' % (args.left_right, args.top, args.bottom))
    args.path_data_file = join(args.dir_data, 'prepared_data.h5') 
    
    args.outChannels = ( args.left_right * 2 + 1 ) * (args.top + args.bottom + 1)
                  
    device = init_env()
        
    test_loader = init_data_loader(args, 'test')
    

    model = PyramidCNN(args.nLevels, args.nPred, args.nPerBlock, 
                        args.nChannels, args.inChannels, args.outChannels, 
                        args.doRes, args.doBN, doELU=False, 
                        predPix=False, predBoxes=False).to(device)
    
    load_weights(args, model)       
    model.eval()
     
    nb = neighbor_connection(*(args.left_right, args.left_right, args.top, args.bottom))
        
    evaluate(model, nb, test_loader, device)
      
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_result', type=str)

    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
        
    parser.add_argument('--nLevels', type=int, default=5)
    parser.add_argument('--nPred', type=int, default=1)
    parser.add_argument('--nPerBlock', type=int, default=2)
    parser.add_argument('--nChannels', type=int, default=80)   
    parser.add_argument('--inChannels', type=int, default=10)
    parser.add_argument('--doRes', type=bool, default=True)
    parser.add_argument('--doBN', type=bool, default=True) 
    
    # neighborhood
    parser.add_argument('--left_right', type=int, default=2)
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--bottom', type=int, default=5)
    
    args = parser.parse_args()
    

    main(args)
    
   
    
    


