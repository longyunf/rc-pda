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
from sklearn.metrics import average_precision_score
import copy

from data_loader_aff import init_data_loader
from rca_utils import depth_to_connect, neighbor_connection, otherHalf, cal_nb_depth
from train_aff import BCE_loss



def load_weights(args, model):
    f_checkpoint = join(args.dir_result, 'checkpoint.tar')
    # f_checkpoint = join(args.dir_result, 'checkpoint_35.tar')        
    if os.path.isfile(f_checkpoint):
        print('load best model')        
        # model.load_state_dict(torch.load(f_checkpoint)['state_dict'])
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



def cal_enhanced_depth(prd_aff, d_radar, nb, thres_aff=0.5):
    '''
        prd_aff: n_nb x h x w
        d_radar: h x w

    '''
    thres_score = thres_aff
    prd = prd_aff
    
    xy_list = nb.xy        
    wt = np.zeros_like(prd)
    dr = np.zeros_like(prd)
    n, h, w = wt.shape
    for i in range(h):
        for j in range(w):
            if d_radar[i,j] > 0:
                for idx, (ox, oy) in enumerate(xy_list):
                    if prd[idx, i, j] > thres_score:
                        xn = j + ox
                        yn = i + oy
                        if xn>= 0 and xn<w and yn>=0 and yn<h:
                            ct = 0
                            while wt[ct,yn,xn] != 0:
                                ct += 1
                            wt[ct,yn,xn] = prd[idx,i,j]
                            dr[ct,yn,xn] = d_radar[i,j]
                            
    wt_sum = np.sum(wt, axis=0)
    msk_blank = wt_sum == 0
    wt_sum[msk_blank] = -1
       
    d_est = np.sum(wt * dr, axis=0) / wt_sum
    d_est[msk_blank] = 0
    
    # idx_max = np.argmax(wt, axis=0)
    # d_est = np.zeros((h,w))
    # for i in range(h):
    #     for j in range(w):
    #         d_est[i,j] = dr[idx_max[i,j], i, j]
       
    return d_est
    

def cal_enhanced_depth_torch(prd_aff, d_radar, nb, device, thres_aff=0.5):
    '''
    use torch for fast implementation
    input:
        prd_aff: tensor: (n_batch, n_nb, h, w) 
        d_radar: tensor: (n_batch, 1, h, w)
    output:
        d_est: numpy (h,w)
    
    '''    
    nb_aff = otherHalf(prd_aff, nb.xy)
    nb2 = copy.deepcopy(nb)
    nb2.reflect()

    nb_depth = cal_nb_depth(d_radar, nb2, device)
    
    nb_aff[nb_aff <= thres_aff] = 0
    nb_aff[nb_depth == 0] = 0
    
    wt_sum = nb_aff.sum(dim=1)
    wt_sum[wt_sum == 0] = -1
    
    d_est = torch.sum(nb_aff * nb_depth, dim = 1) / wt_sum   
    d_est = d_est.squeeze().cpu().numpy()
    
    return d_est


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




        
        
def prd_one_sample(model, nb, test_loader, device, idx = 1, thres_aff = 0.9):
        
    # get output
    with torch.no_grad():
        for ct, sample in enumerate(test_loader):
            if ct == idx:
                data_in, d_lidar, d_radar = sample['data_in'].to(device), sample['d_lidar'].to(device), sample['d_radar'].to(device)               
                # connected = depth_to_connect(d_radar, d_lidar, nb, device)                
                prd = torch.sigmoid( model(data_in)[0] ) 
                
                prd_tensor = prd
                d_radar_tensor = d_radar
                
                im = data_in[0][0:3].permute(1,2,0).to('cpu').numpy()
                # prd_mean_aff = torch.mean(prd[0], dim=0).to('cpu').numpy()
                # gt_mean_aff = torch.mean(connected[0], dim=0).to('cpu').numpy()
                d_lidar = d_lidar[0][0].to('cpu').numpy()
                d_radar = d_radar[0][0].to('cpu').numpy()
                
                prd = prd[0].cpu().numpy()
                
                break
     
   
    d_est = cal_enhanced_depth_torch(prd_tensor, d_radar_tensor, nb, device, thres_aff)
    
    d_est2, wt2 = cal_enhanced_depth_with_max_aff(prd_tensor, d_radar_tensor, nb, device, thres_aff = 0.6)
    
    
    
    
    
    msk = np.logical_and( d_lidar > 0, d_est > 0)
    
    error = np.abs(d_lidar - d_est) * msk
    
    mae = np.mean(error[msk])
    
    print('mae: ', mae)
          
    # visualize output    
    plt.close('all')   
    
    plt.figure()
    plt.imshow(im)
    plt.show()
    
    plt.figure()
    plt.imshow(error, cmap = 'jet')
    plt.colorbar()
    plt.show()
    
    depth = np.concatenate([d_lidar, d_est, d_radar], axis=0)
    plt.figure()
    plt.imshow(depth, cmap = 'jet')
    plt.colorbar()
    plt.show()
    
    
    
    d_radar[0,0]=70
    d_est[0,0]=70
    
    plt_depth_on_im(d_lidar, im, title='lidar depth', ptsSize = 1)
    plt_depth_on_im(d_radar, im, title='Raw radar depth', ptsSize = 1)    
    plt_depth_on_im(d_est, im, title='Enhanced depth', ptsSize = 1)
    
    
    plt_depth_on_im(d_est2, im, title='Enhanced depth2', ptsSize = 1)
    plt_depth_on_im(wt2, im, title='confidence', ptsSize = 1)
    
    plt_depth_on_im(error, im, title='error', ptsSize = 1)
    

    
    
def cal_precision_recall(prd, connected, thres_score = 0.5):
    
    
    msk_prd = prd > thres_score
    msk_connected = connected == 1
    
    tp = np.sum(msk_prd * msk_connected)       # true predicted positive
    det_p = np.sum(msk_prd)                # predicted positive
    all_p = np.sum(msk_connected)          # all positive
    
    return tp, det_p, all_p
    

def evaluate(model, nb, test_loader, device):
    ave_loss = 0 
    ave_ap = 0
    sum_tp = sum_det_p = sum_all_p = 0
      
    thres_score = 0.5
    
    # get output
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
    
    args.left_right, args.top, args.bottom = 3, 20, 5
      
    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result', 'aff_%d_%d_%d' % (args.left_right, args.top, args.bottom))
    args.path_data_file = join(args.dir_data, 'prepared_data_dense.h5') 
    
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
        
    ## prdict one frame
    idx = 50
    thres_aff = 0.95
    prd_one_sample(model, nb, test_loader, device, idx, thres_aff)
        
    # evaluation
    evaluate(model, nb, test_loader, device)
      
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')    
    parser.add_argument('--dir_data', type=str, default='/home/longyunf/media/nuscenes', help='prepared data directory')
    parser.add_argument('--dir_result', type=str, help='directory for training results')

    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    
    
    parser.add_argument('--nLevels', type=int, default=5)
    parser.add_argument('--nPred', type=int, default=1)
    parser.add_argument('--nPerBlock', type=int, default=2)
    parser.add_argument('--nChannels', type=int, default=80)   
    parser.add_argument('--inChannels', type=int, default=10)
    # parser.add_argument('--outChannels', type=int, default=123, help='number of output channel of network; automatically set to 1 if pred_task is foreground_seg')
    parser.add_argument('--doRes', type=bool, default=True)
    parser.add_argument('--doBN', type=bool, default=True) 
    
    # neighborhood
    parser.add_argument('--left_right', type=int, default=2)
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--bottom', type=int, default=5)
    
    args = parser.parse_args()
    

    main(args)
    
   
    
    

