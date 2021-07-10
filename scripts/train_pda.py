
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from os.path import join
from timeit import default_timer as timer
import copy
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from data_loader_pda import init_data_loader
from pyramidNet import PyramidCNN
from pda_utils import depth_to_connect, neighbor_connection



def BCE_loss(aff_prd, lb_aff):
    
    aff_msk = (lb_aff >= 0).float()
    n_pixel = torch.sum(lb_aff>=0).float()    
    n_connect = torch.sum( lb_aff==1 ).float()
       
    w_0 = n_connect / n_pixel
    w_1 = 1 - w_0
    
    weight = aff_msk * ( w_1 * lb_aff + w_0 * ( 1 - lb_aff ) )
          
    criterion = torch.nn.BCEWithLogitsLoss(weight = weight, reduction = 'mean')
    loss_aff = criterion(aff_prd, lb_aff)
    
    return loss_aff


def cal_average_precision(prd, connected):
    msk = connected >= 0            
    prd = torch.sigmoid( prd ) 
    ap = average_precision_score(connected[msk].cpu().numpy(), prd[msk].cpu().numpy())
    return ap



def train(log_interval, model, device, train_loader, optimizer, epoch, nb):
    model.train()   
    ave_loss=0
    
    for batch_idx, sample in enumerate(train_loader):
        data_in, d_lidar, d_radar = sample['data_in'].to(device), sample['d_lidar'].to(device), sample['d_radar'].to(device)
        
        connected = depth_to_connect(d_radar, d_lidar, nb, device)
                
        optimizer.zero_grad() 
        
        prd = model(data_in)[0]                        
        
        loss = BCE_loss(prd, connected)        
        ave_loss += loss.item()

        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_in), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    ave_loss/=len(train_loader)
    print('\nTraining set: Average loss: {:.4f}\n'.format(ave_loss))
    return ave_loss


def test(model, device, test_loader, nb):
    model.eval()
    test_loss = 0
    test_ap = 0
    with torch.no_grad():  
        for sample in tqdm(test_loader, 'Validation'): 
            data_in, d_lidar, d_radar = sample['data_in'].to(device), sample['d_lidar'].to(device), sample['d_radar'].to(device)
            
            connected = depth_to_connect(d_radar, d_lidar, nb, device)
            
            prd = model(data_in)[0] 
            loss = BCE_loss(prd, connected) 
            
            test_ap += cal_average_precision(prd, connected)           
            test_loss += loss.item()
    
    test_loss/= len(test_loader)
    test_ap /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    print('\nTest set: Average precision: {:.4f}\n'.format(test_ap))
    
    return test_loss, test_ap


def save_arguments(args):
    f = open(join(args.dir_result,'args.txt'),'w')
    f.write(repr(args)+'\n')
    f.close()
    

def mkdir(dir1):
    if not os.path.exists(dir1): 
        os.makedirs(dir1)
        print('make directory %s' % dir1)


def init_params(args, model, optimizer):
    loss_train=[]
    loss_val=[]
    
    start_epoch = 1
    state_dict_best = None
    loss_val_min = None
    if args.resume == True:
        f_checkpoint = join(args.dir_result, 'checkpoint.tar')        
        if os.path.isfile(f_checkpoint):
            print('Resume training')
            checkpoint = torch.load(f_checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            loss_train, loss_val = checkpoint['loss']
            loss_val_min = checkpoint['loss_val_min']
            state_dict_best = checkpoint['state_dict_best']
        else:
            print('No checkpoint file is found.')
                        
    return loss_train, loss_val, start_epoch, state_dict_best, loss_val_min



def save_checkpoint(epoch, model, optimizer, loss_train, loss_val, loss_val_min, state_dict_best, args):
    if epoch == 1:
        loss_val_min = loss_val[-1]
        state_dict_best = copy.deepcopy( model.state_dict() )
    elif loss_val[-1] < loss_val_min:
        loss_val_min = loss_val[-1]
        state_dict_best = copy.deepcopy( model.state_dict() )
            
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss': [loss_train, loss_val],
             'loss_val_min': loss_val_min,
             'state_dict_best': state_dict_best}
    
    torch.save(state, join(args.dir_result, 'checkpoint.tar'))
    if epoch % 5 == 0:
        torch.save(state, join(args.dir_result, 'checkpoint_%d.tar' % epoch))
        
    return loss_val_min, state_dict_best
    

def plot_and_save_loss_curve(epoch, loss_train, loss_val):
    plt.close('all')
    plt.figure()  
    t=np.arange(1,epoch+1)
    plt.plot(t,loss_train,'b.-')
    plt.plot(t,loss_val,'r.-')
    plt.grid()
    plt.legend(['training loss','testing loss'],loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')   
    plt.title('loss in logscale')
    plt.savefig(join(args.dir_result, 'loss.png'))


def init_env():
    torch.manual_seed(args.seed)    
    use_cuda = torch.cuda.is_available()    
    device = torch.device("cuda" if use_cuda else "cpu")    
    cudnn.benchmark = True if use_cuda else False
    
    return device
    
     
def main(args):
    args.outChannels = ( args.left_right * 2 + 1 ) * (args.top + args.bottom + 1)
    print('output channels: ', args.outChannels)
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
        
    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result', 'pda_%d_%d_%d' % (args.left_right, args.top, args.bottom))              
    mkdir(args.dir_result)  
    
    args.dir_summary = join(args.dir_result, 'summary') 
    mkdir(args.dir_summary)     
      
    args.path_data_file = join(args.dir_data, 'prepared_data.h5') 
    save_arguments(args)    
        
    
    device = init_env()
    
    writer = SummaryWriter(args.dir_summary, flush_secs=30)
            
    model = PyramidCNN(args.nLevels, args.nPred, args.nPerBlock, 
                        args.nChannels, args.inChannels, args.outChannels, 
                        args.doRes, args.doBN, doELU=False, 
                        predPix=False, predBoxes=False).to(device)


    optimizer = torch.optim.RMSprop(model.parameters(),
                                 	lr = args.lr, 
                                 	weight_decay = 0, 
                                 	momentum = args.momentum)
    
    
    loss_train, loss_val, start_epoch, state_dict_best, loss_val_min = \
    init_params(args, model, optimizer)
    
    
    train_loader = init_data_loader(args, 'train')
    val_loader = init_data_loader(args, 'val')
    
    
    nb = neighbor_connection(*(args.left_right, args.left_right, args.top, args.bottom))
    
  
    for epoch in range(start_epoch, args.epochs + 1):
        start = timer()
        
        loss_train.append(train(args.log_interval, model, device, train_loader, optimizer, epoch, nb))
        loss_val_epoch, ap = test(model, device, val_loader, nb)
        
        loss_val.append(loss_val_epoch)
        writer.add_scalar('val_ap', ap, epoch)
        writer.flush()
        
        loss_val_min, state_dict_best = save_checkpoint(epoch, model, optimizer, loss_train, loss_val, loss_val_min, state_dict_best, args)
        plot_and_save_loss_curve(epoch, loss_train, loss_val) 
          
        end = timer(); t = (end - start) / 60; print('Time used: %.1f minutes\n' % t)
           

    if args.do_test:
        test_loader = init_data_loader(args, 'test')      
        f_checkpoint = join(args.dir_result, 'checkpoint.tar')        
        if os.path.isfile(f_checkpoint):
            print('load best model')
            checkpoint = torch.load(f_checkpoint)
            model.load_state_dict(checkpoint['state_dict_best'])        
        loss_test, ap = test(model, device, test_loader, nb)
        print('testing loss:', loss_test)
        print('\naverage_precision:', ap)
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')    
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_result', type=str)
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)    
    parser.add_argument('--num_workers', type=int, default=0) 
    parser.add_argument('--no_data_shuffle', type=bool, default=False, help='for generating training data in order')
    
    parser.add_argument('--nLevels', type=int, default=5)
    parser.add_argument('--nPred', type=int, default=1)
    parser.add_argument('--nPerBlock', type=int, default=2)
    parser.add_argument('--nChannels', type=int, default=80)   
    parser.add_argument('--inChannels', type=int, default=10)
    parser.add_argument('--doRes', type=bool, default=True)
    parser.add_argument('--doBN', type=bool, default=True)        
    parser.add_argument('--do_test', type=bool, default=True, help='compute loss for testing set')
    
    # neighborhood
    parser.add_argument('--left_right', type=int, default=2)
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--bottom', type=int, default=5)
        
    args = parser.parse_args()
    main(args)



