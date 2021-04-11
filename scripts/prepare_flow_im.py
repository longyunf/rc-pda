'''
Extract images

'''

import skimage.io as io
import os
from os.path import join
import glob
import argparse
from skimage.transform import resize
import torch
import numpy as np

from nuscenes.nuscenes import NuScenes


def downsample_im(im, downsample_scale, y_cutoff):
    h_im, w_im = im.shape[0:2]        
    h_im = int( h_im / downsample_scale )
    w_im = int( w_im / downsample_scale ) 
    
    im = resize(im, (h_im,w_im,3), order=1, preserve_range=True, anti_aliasing=False) 
    im = im.astype('uint8')
    im = im[y_cutoff:,...]
    return im
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
        
    args = parser.parse_args()

    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '..', 'data')
    dir_nuscenes = os.path.join(args.dir_data, 'nuscenes')
    
    
    downsample_scale = 4
    y_cutoff = 33
    
       
    nusc = NuScenes(args.version, dataroot = dir_nuscenes, verbose=False)
       
    dir_data_out = join(args.dir_data, 'prepared_data')
    if not os.path.exists(dir_data_out):
        os.makedirs(dir_data_out)
    
    'remove all files in the output folder'
    f_list=glob.glob(join(dir_data_out,'*'))
    for f in f_list:
        os.remove(f)
    print('removed %d old files in output folder' % len(f_list))
    
    
    sample_indices = torch.load(join(args.dir_data,'data_split.tar'))['all_indices'] 
         
    ct = 0         
    for sample_idx in sample_indices:

        cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        
        if cam_data['next']:
                
            cam_token2 = cam_data['next']                    
            cam_data2 = nusc.get('sample_data', cam_token2)
            cam_path2 = join(nusc.dataroot, cam_data2['filename'])
            im2 = io.imread(cam_path2)
            
            cam_token3 = cam_data2['next']        
            cam_data3 = nusc.get('sample_data', cam_token3)
            cam_path3 = join(nusc.dataroot, cam_data3['filename'])
            im3 = io.imread(cam_path3)
             
            im = downsample_im(im2, downsample_scale, y_cutoff)
            im_next = downsample_im(im3, downsample_scale, y_cutoff)
                       
            io.imsave(join(dir_data_out, '%05d_im.jpg' % sample_idx), im)
            io.imsave(join(dir_data_out, '%05d_im_next.jpg' % sample_idx), im_next)          
           
        ct += 1
        print('Save image %d/%d' % ( ct, len(sample_indices) ) )
        

    
    