'''
Accumulate Lidar points based on ego-vehicle pose and GT bounding boxes.
  
'''

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from functools import reduce
from pyquaternion import Quaternion
import skimage.io as io
import os
from os.path import join
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import argparse
from skimage.transform import resize
import copy


def merge_pc(pc1, pc2):

    points = np.zeros((pc1.nbr_dims(), 0))
    all_pc = LidarPointCloud(points)    
    all_pc.points = np.hstack((all_pc.points, pc1.points, pc2.points))
    
    return all_pc


def merge_lidar(nusc, sample_idx, n_backward, n_forward, box_tracks, n_skip=0):
    '''
    Merge n_backward and n_forward Lidar frames with current key Lidar frame based on bounding boxes and ego-pose  
    
    inputs:
        box_tracks: a dictionary of bounding boxes poses at time of each frame
    
    outputs: 
       x1, y1, x2, y2: projected coordinates on image
       depth1, depth2: depth map for im1 and im2
        
    '''
    
    def get_height_mask(pc, car_from_current, h_min=0.3, h_max=3):
        '''
        get mask for Lidar points based on their height in vehicle coordinates (to select certain height range)
        '''
        pc_temp = copy.deepcopy(pc)
        pc_temp.transform(car_from_current)
        pts = pc_temp.points
        msk = np.logical_and(pts[2]>=h_min, pts[2]<=h_max)
        
        return msk
    
    
    def cal_box_corners_in_global(nusc, ref_sd_token, box_tracks):
        corner_pc_all = LidarPointCloud(np.zeros((4,0)))
        
        for instance_token in box_tracks[ref_sd_token]:            
            box = box_tracks[ref_sd_token][instance_token]
 
            if 'vehicle' in box['category_name']:
                M_object_to_global = transform_matrix(box['translation'],
                                                      Quaternion(box['rotation']), inverse=False)  # rotation + tranlation
                w,l,h = box['size']      
                pts_c = []
                for xc in [-l/2, l/2]:
                    for yc in [-w/2, w/2]:
                        for zc in [-h/2, h/2]:
                            pts_c.append([xc,yc,zc,1])
                
                pts_c = np.array(pts_c).T                
                corner_pc = LidarPointCloud(pts_c)           
                corner_pc.transform(M_object_to_global)
                
                corner_pc_all.points = np.hstack((corner_pc_all.points, corner_pc.points))
        
        return corner_pc_all
    
    
    def current_to_global_at_ref_time(nusc, current_sd_token, ref_sd_token, box_tracks):
        ''' 
        transform Lidar points of current frame to global coordinates at the time of the reference frame
               
        msk_in: msk for points in bounding boxes
        
        '''        
        current_sd_rec = nusc.get('sample_data', current_sd_token)        
        pc = LidarPointCloud(np.zeros((4,0)))
        current_pc = pc.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        min_distance = 2.5
        current_pc.remove_close(min_distance)
        
        msk_mv = np.zeros(current_pc.nbr_points())
                
        # Get pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)
        
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)
        
        # low height 0.3 - 2 m
        msk_low_h = get_height_mask(current_pc, car_from_current, h_min=0.3, h_max=2)
        current_pc.points[3,:] = msk_low_h
        
        M_global_from_current = reduce(np.dot, [global_from_car, car_from_current])
        current_pc.transform(M_global_from_current)        
        
        obj_pc_all = LidarPointCloud(np.zeros((4,0)))
        msk_in_box_all = np.zeros(0)
        
        for instance_token in box_tracks[ref_sd_token]:
            if instance_token in box_tracks[current_sd_token].keys():
                box = box_tracks[current_sd_token][instance_token]
                box_aim = box_tracks[ref_sd_token][instance_token]
     
                M_global_to_object = transform_matrix(box['translation'],
                                                      Quaternion(box['rotation']), inverse=True)  # -translation + -rotation       
                M_object_to_global = transform_matrix(box['translation'],
                                                      Quaternion(box['rotation']), inverse=False)  # rotation + tranlation
                
                current_pc.transform(M_global_to_object)
                                 
                w,l,h = box['size']
                
                idx_obj = current_pc.points[0] > -l/2
                idx_obj = np.logical_and(idx_obj, current_pc.points[0] < l/2)
                idx_obj = np.logical_and(idx_obj, current_pc.points[1] > -w/2)
                idx_obj = np.logical_and(idx_obj, current_pc.points[1] < w/2)
                idx_obj = np.logical_and(idx_obj, current_pc.points[2] > -h/2)
                idx_obj = np.logical_and(idx_obj, current_pc.points[2] < h/2)                
                
                obj_pc = LidarPointCloud( copy.deepcopy( current_pc.points[:,idx_obj] ) )
                
                idx_other = np.logical_not(idx_obj)            
                current_pc.points = current_pc.points[:,idx_other]            
                current_pc.transform(M_object_to_global)
                
                M_object_to_global_aim = transform_matrix(box_aim['translation'], 
                                                     Quaternion(box_aim['rotation']), inverse=False)
                            
                obj_pc.transform(M_object_to_global_aim)
                obj_pc_all.points = np.hstack((obj_pc_all.points, obj_pc.points))
                                
                msk_in_box = msk_mv[idx_obj]
                msk_mv = msk_mv[idx_other]
                if 'vehicle' in box['category_name']:
                    msk_in_box = np.ones(len(msk_in_box))
                msk_in_box_all = np.concatenate( ( msk_in_box_all, msk_in_box), axis=0)
                                      
        current_pc.points = np.hstack((current_pc.points, obj_pc_all.points))
        msk_in_box = np.concatenate( ( msk_mv, msk_in_box_all), axis=0)
        
        return current_pc, msk_in_box

    
    def cal_matrix_refCam_from_global(cam_data):
        ref_pose_rec = nusc.get('ego_pose', cam_data['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])    
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)        
        M_ref_from_global = reduce(np.dot, [ref_from_car, car_from_global])        
        return M_ref_from_global
    
     
    def proj2im(pc_cam, cam_data, min_z = 2):        
        cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])         
        depth = pc_cam.points[2]    
        msk = pc_cam.points[2] >= min_z       
        points = view_points(pc_cam.points[:3, :], np.array(cs_rec['camera_intrinsic']), normalize=True)        
        x, y = points[0], points[1]
        msk =  reduce(np.logical_and, [x>0, x<1600, y>0, y<900, msk])        
        return x, y, depth, msk
        
        
  
    sample_rec = nusc.sample[sample_idx]
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    
    cam_token = sample_rec['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)   
    cam_token1 = cam_data['next']                    
    cam_data1 = nusc.get('sample_data', cam_token1)
    cam_token2 = cam_data1['next'] 
    cam_data2 = nusc.get('sample_data', cam_token2)    
    
    all_pc1 = LidarPointCloud(np.zeros((4,0)))
    all_pc2 = LidarPointCloud(np.zeros((4,0)))
    
    all_msk_in = np.zeros(0)   # points in bounding box
            
    M_refCam1_from_global = cal_matrix_refCam_from_global(cam_data1)
    M_refCam2_from_global = cal_matrix_refCam_from_global(cam_data2)
    
    
    ct_forward = 0  
    current_sd_token = ref_sd_token        
      
    
    corner_pc = cal_box_corners_in_global(nusc, cam_token1, box_tracks)
    corner_pc.transform(M_refCam2_from_global)
    
    while ct_forward <= n_forward and current_sd_token != '': 

        current_pc1, msk_in1 = current_to_global_at_ref_time(nusc, current_sd_token, cam_token1, box_tracks)
        current_pc2, _       = current_to_global_at_ref_time(nusc, current_sd_token, cam_token2, box_tracks)
        
        current_pc1.transform(M_refCam1_from_global)
        current_pc2.transform(M_refCam2_from_global)       
                
        # Merge
        all_pc1.points = np.hstack((all_pc1.points, current_pc1.points))
        all_pc2.points = np.hstack((all_pc2.points, current_pc2.points))
        
        all_msk_in = np.concatenate([all_msk_in, msk_in1], axis=0)
        
                      
        for _ in range(n_skip + 1):
            current_sd_rec = nusc.get('sample_data', current_sd_token)
            current_sd_token = current_sd_rec['next']
            ct_forward += 1        
            if current_sd_token == '':
                break
    

    ct_backward = 0 
    current_sd_token = ref_sd_token        
          
    for _ in range(n_skip + 1):
        current_sd_rec = nusc.get('sample_data', current_sd_token)
        current_sd_token = current_sd_rec['prev']
        ct_backward += 1        
        if current_sd_token == '':
            break   
    
    while ct_backward <= n_backward and current_sd_token != '': 
        
        current_pc1, msk_in1 = current_to_global_at_ref_time(nusc, current_sd_token, cam_token1, box_tracks)
        current_pc2, _       = current_to_global_at_ref_time(nusc, current_sd_token, cam_token2, box_tracks)

        current_pc1.transform(M_refCam1_from_global)
        current_pc2.transform(M_refCam2_from_global)
        
        # Merge
        all_pc1.points = np.hstack((all_pc1.points, current_pc1.points))
        all_pc2.points = np.hstack((all_pc2.points, current_pc2.points))
        
        all_msk_in = np.concatenate([all_msk_in, msk_in1], axis=0)
                 
        for _ in range(n_skip + 1):
            current_sd_rec = nusc.get('sample_data', current_sd_token)
            current_sd_token = current_sd_rec['prev']
            ct_backward += 1        
            if current_sd_token == '':
                break
    
    msk_low_h = all_pc1.points[3,:].astype(bool)
    
    x1, y1, depth1, msk1 = proj2im(all_pc1, cam_data1)
    x2, y2, depth2, msk2 = proj2im(all_pc2, cam_data2) 
    
    # box corners
    x_cn, y_cn, depth_cn, msk_cn = proj2im(corner_pc, cam_data1) 
            
    msk = msk1 * msk2
    
    x1, y1, x2, y2, depth1, depth2 = x1[msk], y1[msk], x2[msk], y2[msk], depth1[msk], depth2[msk]
    msk_low_h = msk_low_h[msk]
    msk_in = all_msk_in[msk]
                   
    return x1, y1, depth1, x2, y2, depth2, msk_low_h, msk_in, x_cn, y_cn, depth_cn, msk_cn

    

def cal_depthMap_flow(x1, y1, depth1, x2, y2, depth2, msk_low_h, msk_mv, downsample_scale, y_cutoff):
    
    h_im, w_im = 900, 1600
    h_new = int( h_im / downsample_scale )
    w_new = int( w_im / downsample_scale ) 
      
    depth_map1 = np.zeros( (h_new, w_new) , dtype=float)
    msk_map_low_h = np.zeros( (h_new, w_new) , dtype=bool)    # mask for region 0.3-3m in car coordinates    
    msk_map_mv = np.zeros( (h_new, w_new) , dtype=bool)       # mask for moving objects
    flow = np.zeros( (h_new, w_new, 2) , dtype=float)         # dx, dy
        
    # pixel square model
    x1 = (x1 + 0.5) / downsample_scale - 0.5
    y1 = (y1 + 0.5) / downsample_scale - 0.5
    x2 = (x2 + 0.5) / downsample_scale - 0.5
    y2 = (y2 + 0.5) / downsample_scale - 0.5
    
    x1 = np.clip(x1, 0, w_new - 1)
    x2 = np.clip(x2, 0, w_new - 1)
    y1 = np.clip(y1, 0, h_new - 1)
    y2 = np.clip(y2, 0, h_new - 1)
    
 
    for i in range(len(x1)):
        x1_one, y1_one = int(round( x1[i] )), int(round( y1[i] ))
        # x2_one, y2_one = int(round( x2[i] )), int(round( y2[i] ))
        
        if depth_map1[y1_one,x1_one] == 0:
            depth_map1[y1_one,x1_one] = depth1[i]            
            flow[y1_one, x1_one, ...] = [x2[i] - x1[i], y2[i] - y1[i]] 
            msk_map_low_h[y1_one, x1_one] = msk_low_h[i] 
            msk_map_mv[y1_one, x1_one] = msk_mv[i]                            
        elif depth_map1[y1_one,x1_one] > depth1[i]:                           
            depth_map1[y1_one,x1_one] = depth1[i]            
            flow[y1_one, x1_one, ...] = [x2[i] - x1[i], y2[i] - y1[i]]
            msk_map_low_h[y1_one, x1_one] = msk_low_h[i]
            msk_map_mv[y1_one, x1_one] = msk_mv[i]
      
    depth_map1, flow = depth_map1[y_cutoff:,...], flow[y_cutoff:,...]
    msk_map_low_h = msk_map_low_h[y_cutoff:,...]
    msk_map_mv = msk_map_mv[y_cutoff:,...]
               
    return depth_map1, flow, msk_map_low_h, msk_map_mv
           
    
def lidar_pts(nusc, sample_idx):
    lidar_token = nusc.sample[sample_idx]['data']['LIDAR_TOP']        
    pointsensor1 = nusc.get('sample_data', lidar_token)
        
    pcl_path = os.path.join(nusc.dataroot, pointsensor1['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    
    return pc.points[:3]


def load_im(nusc, sample_idx):
    
    cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    
    cam_token2 = cam_data['next']                    
    cam_data2 = nusc.get('sample_data', cam_token2)
    cam_path2 = join(nusc.dataroot, cam_data2['filename'])
    im1 = io.imread(cam_path2)
    
    cam_token3 = cam_data2['next']        
    cam_data3 = nusc.get('sample_data', cam_token3)
    cam_path3 = join(nusc.dataroot, cam_data3['filename'])
    im2 = io.imread(cam_path3)
    
    return im1, im2
 


def get_tracks_forward(nusc, sample_idx):
    '''
    Estimate bounding boxes of objects in current and next key Lidar frame as well as intermediate Lidar frames and 2nd and 3rd image frames
        
    return:
        track_forward_key: tracks for key frames; dictionary; key: lidar sample_data token; value: a dictionary of bounding box poses (instance_key: boxes)
        track_forward_non: tracks for intermediate frames
    '''
     
    sample = nusc.sample[sample_idx]   
    sample_next = nusc.sample[sample_idx + 1]
    
    track_forward_key = {}              # bounding boxes on key frames (with ground truth labels)
    track_forward_non = {}              # bounding boxes on intermediate frames
    
    sample_sd_token = sample['data']['LIDAR_TOP']
    sample_next_sd_token = sample_next['data']['LIDAR_TOP']
        
    track_forward_key[sample_sd_token] = {}
    track_forward_key[sample_next_sd_token] = {}
        
    sd_token = sample_sd_token
    while True:        
        sd_rec = nusc.get('sample_data', sd_token)
        sd_token = sd_rec['next']
        if sd_token == sample_next_sd_token:
            break
        track_forward_non[sd_token] = {}
    
    cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)    
    cam_token2 = cam_data['next']                    
    cam_data2 = nusc.get('sample_data', cam_token2)    
    cam_token3 = cam_data2['next']  
    
    track_forward_non[cam_token2] = {}
    track_forward_non[cam_token3] = {}  


    def judge_moving(ann, ann_next, thres_dist):
        trans1 = ann['translation']
        trans2 = ann_next['translation']
        dist = ( (trans1[0] - trans2[0])**2 + (trans1[1] - trans2[1])**2 ) ** 0.5
        if dist > thres_dist:
            is_moving = True
        else:
            is_moving = False            
        return is_moving
        
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['next']:     # the same instance appears in two key sample frames
            ann_next = nusc.get('sample_annotation', ann['next'])            
            instance_token = ann['instance_token']
            instance_pose = { k : ann[k] for k in ['translation', 'size', 'rotation', 'instance_token', 'category_name'] }
            instance_pose_next = { k : ann_next[k] for k in ['translation','size', 'rotation', 'instance_token', 'category_name'] }            
            
            is_moving = judge_moving(ann, ann_next, thres_dist=0.2)           
            instance_pose['is_moving'], instance_pose_next['is_moving'] = is_moving, is_moving
                        
            track_forward_key[sample_sd_token][instance_token] = instance_pose
            track_forward_key[sample_next_sd_token][instance_token] = instance_pose_next
    
                
    for key in track_forward_key[sample_sd_token]:
        
        box = track_forward_key[sample_sd_token][key]
        box_next = track_forward_key[sample_next_sd_token][key]
        
        qu = Quaternion(*box['rotation'])
        qu_next = Quaternion(*box_next['rotation'])
        
        d_trans = np.array(box_next['translation']) - np.array(box['translation'])
        
        t = nusc.get('sample_data', sample_sd_token)['timestamp']
        t_next = nusc.get('sample_data', sample_next_sd_token)['timestamp']
        
        for sd_token in track_forward_non:           
            t_sd = nusc.get('sample_data', sd_token)['timestamp']
            alpha_t = (t_sd - t)/(t_next - t)
            
            trans_sd = list( np.array(box['translation']) + alpha_t * d_trans )                                              
            angle_sd_q = list( Quaternion.slerp(qu, qu_next, amount = alpha_t).elements )
            
            track_forward_non[sd_token][box['instance_token']] = {'translation': trans_sd, 'size':box['size'], 'rotation': angle_sd_q, 'instance_token': box['instance_token'], 'category_name': box['category_name'],'is_moving': box['is_moving'] }
        

    return track_forward_key, track_forward_non
        


def update_key_tracks(track_old, track_new):
    for sd_token in track_new:
        if sd_token in track_old.keys():
            track_old[sd_token].update(track_new[sd_token])
        else:
            track_old[sd_token] = track_new[sd_token]
            
    return track_old
    
    
def update_non_key_tracks(track_old, track_new):   
    track_old.update(track_new)
    
    return track_old


def downsample_im(im, downsample_scale, y_cutoff):
    h_im, w_im = im.shape[0:2]        
    h_im = int( h_im / downsample_scale )
    w_im = int( w_im / downsample_scale ) 
    
    im = resize(im, (h_im,w_im,3), order=1, preserve_range=True, anti_aliasing=False) 
    im = im.astype('uint8')
    im = im[y_cutoff:,...]
    return im


def plt_depth_on_im(depth_map, im, title=''):
    
    h,w = im.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    msk = depth_map > 0
    
    depth_map[0,0] = 0.1
    depth_map[depth_map>70] = 70
    title=''
        
    plt.figure()
    plt.imshow(im) 
    plt.scatter(x_map[msk], y_map[msk], c=depth_map[msk], s=1, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    
    
def plt_flow_on_im(flow, im, skip = 3, title=''):
    
    h,w = im.shape[0:2] 
    plt.figure()
    plt.imshow(im)
    msk1 = flow[:,:,0] != 0
    msk2 = flow[:,:,1] != 0
    msk = np.logical_or(msk1,msk2)
    
    for i in range(0, h, skip+1):
        for j in range(0, w, skip+1):
            if msk[i,j]:
                plt.arrow(j,i, flow[i,j,0], flow[i,j,1], length_includes_head=True, width=0.05, head_width=0.5, color='cyan')
    plt.title(title)
     
     

def flow_l2_error(flow_lidar, flow_im):
    msk1 = flow_lidar[:,:,0] != 0
    msk2 = flow_lidar[:,:,1] != 0
    msk = np.logical_or(msk1,msk2)
    
    error = flow_lidar - flow_im
    l2 = (error[...,0] ** 2 + error[...,1] ** 2) ** 0.5
    l2 = l2 * msk
    
    return l2


def filter_occlusion(depth_map, msk_map_low_h, msk_map_mv, flow_lidar, flow_im, thres=4):
    msk = depth_map > 0
    
    error = flow_lidar - flow_im
    l2 = (error[...,0] ** 2 + error[...,1] ** 2) ** 0.5
    
    msk_occ2 = np.logical_and(msk, l2 > thres)    
    y_list, x_list = np.where(msk_occ2==True)
        
    depth_map[msk_occ2] = 0
    msk_map_low_h[msk_occ2] = False
    msk_map_mv[msk_occ2] = False
    flow_lidar[y_list, x_list, :] = [0,0]
    
    return depth_map, msk_map_low_h, msk_map_mv, flow_lidar
    

def lidarFlow2uv(flow, K, depth_map, downsample_scale=4, y_cutoff=33):
    '''
    uv_map: h x w x 2
    '''
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    
    h,w = flow.shape[:2]
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map, y_map = x_map.astype('float32'), y_map.astype('float32')
    x_map += flow[..., 0]
    y_map += flow[..., 1]
    
    cx = cx / downsample_scale
    cy = cy / downsample_scale - y_cutoff
    f = f / downsample_scale
    
    u_map = (x_map - cx) / f
    v_map = (y_map - cy) / f
        
    msk = depth_map > 0
    u_map *= msk
    v_map *= msk
        
    uv_map = np.stack([u_map,v_map], axis=2)
    
    return uv_map

 
   
def cal_msk_bbox(x_cn, y_cn, msk_cn, downsample_scale, y_cutoff):
       '''
       msk_cn: mask for corners in image          
       '''          
       h_im, w_im = 900, 1600
       h_new = int( h_im / downsample_scale )
       w_new = int( w_im / downsample_scale ) 
         
       msk_map_mv = np.zeros( (h_new, w_new) , dtype=bool)   
                
       # pixel square model
       x_cn = (x_cn + 0.5) / downsample_scale - 0.5
       y_cn = (y_cn + 0.5) / downsample_scale - 0.5
              
       for start in range(0, len(x_cn, ), 8):
           x_list = x_cn[start: start + 8]
           y_list = y_cn[start: start + 8]
           msk_list = msk_cn[start: start + 8]
           
           x_list_in, y_list_in = x_list[msk_list], y_list[msk_list] 

           if len(x_list_in)!=0 and len(y_list_in)!=0:
               x_list, y_list = np.clip(x_list, 0, w_new - 1), np.clip(y_list, 0, h_new - 1)               
               x_min, x_max, y_min, y_max = int(round(np.min(x_list))), int(round(np.max(x_list))), int(round(np.min(y_list))), int(round(np.max(y_list)))               
               msk_map_mv[y_min : y_max+1, x_min : x_max+1] = True
           
       msk_map_mv = msk_map_mv[y_cutoff:,...]
           
       return msk_map_mv
   
 
     
def filter_occlusion_by_bbox(depth_map, seg, msk_map_in, x_cn, y_cn, depth_cn, msk_cn, downsample_scale, y_cutoff):
  
    '''
    depth_map: depth before filtering
    seg: vehicle semantic segmenation
    msk_map_in: mask for points in vehicle bounding box
    x_cn, y_cn: projected bbox corner coordinates in image        
    depth_cn: depth of corners
    msk_cn: mask for corners in the field of view of image
       
    '''          
    h_im, w_im = 900, 1600
    h_new = int( h_im / downsample_scale )
    w_new = int( w_im / downsample_scale ) 
             
    # pixel square model
    x_cn = (x_cn + 0.5) / downsample_scale - 0.5
    y_cn = (y_cn + 0.5) / downsample_scale - 0.5
    
    y_cn = y_cn - y_cutoff
    h_new = h_new - y_cutoff
           
    for start in range(0, len(x_cn, ), 8):
        x_list = x_cn[start: start + 8]
        y_list = y_cn[start: start + 8]
        msk_list = msk_cn[start: start + 8]
        d_list = depth_cn[start: start + 8]
        
        x_list, y_list = x_list[msk_list], y_list[msk_list] 

        if len(x_list)!=0 and len(y_list)!=0:
            d_max = np.max(d_list)
            x_list, y_list = np.clip(x_list, 0, w_new - 1), np.clip(y_list, 0, h_new - 1)               
            
            x_min, x_max, y_min, y_max = int(round(np.min(x_list))), int(round(np.max(x_list))), int(round(np.min(y_list))), int(round(np.max(y_list)))               
            
            for i in range(y_min, y_max + 1):
                for j in range(x_min, x_max + 1):
                    if seg[i,j] and ( depth_map[i,j] > 0 ) and ( msk_map_in[i,j]==False ) and depth_map[i,j] > d_max:
                        depth_map[i,j] = 0                            
    msk = depth_map > 0
        
    return depth_map, msk


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

   

   