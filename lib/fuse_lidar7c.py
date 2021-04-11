'''

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

from mpl_toolkits.mplot3d import Axes3D
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
        box_tracks: a dictionary finding bounding boxes poses at time of each frame
    
    outputs: 
       x1, y1, x2, y2: projected coordinates on image
       depth1, depth2: depth map for im1 and im2
        
    '''
    
    def get_height_mask(pc, car_from_current, h_min=0.3, h_max=3):
        '''
        get mask for Lidar points based on their height in vehicle coordinates (to select certain height range and remove ground points)
        '''
        pc_temp = copy.deepcopy(pc)
        pc_temp.transform(car_from_current)
        pts = pc_temp.points
        msk = np.logical_and(pts[2]>=h_min, pts[2]<=h_max)
        
        return msk
    
    
    
    def current_to_global_at_ref_time(nusc, current_sd_token, ref_sd_token, box_tracks):
        ''' 
        transform Lidar points of current frame to global coordinates at the time of the reference frame
        
        '''        
        current_sd_rec = nusc.get('sample_data', current_sd_token)        
        pc = LidarPointCloud(np.zeros((4,0)))
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = pc.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        min_distance = 2.5     # let minimal distance be 2 to remove hits on ego vehicle
        current_pc.remove_close(min_distance)
        

        # Get pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)
        
        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)
        
        msk_low_h = get_height_mask(current_pc, car_from_current, h_min=0.3, h_max=3)
        current_pc.points[3,:] = msk_low_h
        
        M_global_from_current = reduce(np.dot, [global_from_car, car_from_current])
        current_pc.transform(M_global_from_current)        
        
        obj_pc_all = LidarPointCloud(np.zeros((4,0)))
        
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
                
                obj_pc = LidarPointCloud(current_pc.points[:,idx_obj])
                
                idx_other = np.logical_not(idx_obj)            
                current_pc.points = current_pc.points[:,idx_other]            
                current_pc.transform(M_object_to_global)
                
                M_object_to_global_aim = transform_matrix(box_aim['translation'], 
                                                     Quaternion(box_aim['rotation']), inverse=False)  # rotation + tranlation
                            
                obj_pc.transform(M_object_to_global_aim)
                obj_pc_all.points = np.hstack((obj_pc_all.points, obj_pc.points))
                
        current_pc.points = np.hstack((current_pc.points, obj_pc_all.points))
        
        return current_pc
    
    def cal_matrix_refCam_from_global(cam_data):
        ref_pose_rec = nusc.get('ego_pose', cam_data['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])    
        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
        # Homogeneous transformation matrix from global to current ego car frame.
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
    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    
    cam_token = sample_rec['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)   
    cam_token1 = cam_data['next']                    
    cam_data1 = nusc.get('sample_data', cam_token1)
    cam_token2 = cam_data1['next'] 
    cam_data2 = nusc.get('sample_data', cam_token2)    
    
    all_pc1 = LidarPointCloud(np.zeros((4,0)))
    all_pc2 = LidarPointCloud(np.zeros((4,0)))
        
    M_refCam1_from_global = cal_matrix_refCam_from_global(cam_data1)
    M_refCam2_from_global = cal_matrix_refCam_from_global(cam_data2)
    
    
    ct_forward = 0  
    current_sd_token = ref_sd_token        
      
          
    while ct_forward <= n_forward and current_sd_token != '': 

        current_pc1 = current_to_global_at_ref_time(nusc, current_sd_token, cam_token1, box_tracks)
        current_pc2 = current_to_global_at_ref_time(nusc, current_sd_token, cam_token2, box_tracks)
        
        current_pc1.transform(M_refCam1_from_global)
        current_pc2.transform(M_refCam2_from_global)
        
        # Merge
        all_pc1.points = np.hstack((all_pc1.points, current_pc1.points))
        all_pc2.points = np.hstack((all_pc2.points, current_pc2.points))
                      
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
        
        current_pc1 = current_to_global_at_ref_time(nusc, current_sd_token, cam_token1, box_tracks)
        current_pc2 = current_to_global_at_ref_time(nusc, current_sd_token, cam_token2, box_tracks)

        current_pc1.transform(M_refCam1_from_global)
        current_pc2.transform(M_refCam2_from_global)
        
        # Merge
        all_pc1.points = np.hstack((all_pc1.points, current_pc1.points))
        all_pc2.points = np.hstack((all_pc2.points, current_pc2.points))
                 
        for _ in range(n_skip + 1):
            current_sd_rec = nusc.get('sample_data', current_sd_token)
            current_sd_token = current_sd_rec['prev']
            ct_backward += 1        
            if current_sd_token == '':
                break
    
    msk_low_h = all_pc1.points[3,:].astype(bool)
    
    x1, y1, depth1, msk1 = proj2im(all_pc1, cam_data1)
    x2, y2, depth2, msk2 = proj2im(all_pc2, cam_data2) 
    
    msk = msk1 * msk2
    
    x1, y1, x2, y2, depth1, depth2 = x1[msk], y1[msk], x2[msk], y2[msk], depth1[msk], depth2[msk]
    msk_low_h = msk_low_h[msk]
                 
    return x1, y1, depth1, x2, y2, depth2, msk_low_h

    

def cal_depthMap_flow(x1, y1, depth1, x2, y2, depth2, msk_low_h, downsample_scale, y_cutoff):
    
    h_im, w_im = 900, 1600
    h_new = int( h_im / downsample_scale )
    w_new = int( w_im / downsample_scale ) 
      
    depth_map1 = np.zeros( (h_new, w_new) , dtype=float)
    msk_map_low_h = np.zeros( (h_new, w_new) , dtype=bool)    # mask for region 0.3-3m in car coordinates    
    flow = np.zeros( (h_new, w_new, 2) , dtype=float)      # dx, dy
        
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
        elif depth_map1[y1_one,x1_one] > depth1[i]:                           
            depth_map1[y1_one,x1_one] = depth1[i]            
            flow[y1_one, x1_one, ...] = [x2[i] - x1[i], y2[i] - y1[i]]
            msk_map_low_h[y1_one, x1_one] = msk_low_h[i]
      
    depth_map1, flow = depth_map1[y_cutoff:,...], flow[y_cutoff:,...]
    msk_map_low_h = msk_map_low_h[y_cutoff:,...]
               
    return depth_map1, flow, msk_map_low_h
           
    
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
    
    sample_sd_token = sample['data']['LIDAR_TOP']               # sample data token
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
         
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['next']:     # the same instance appears in two key sample frames
            ann_next = nusc.get('sample_annotation', ann['next'])            
            instance_token = ann['instance_token']
            instance_pose = { k : ann[k] for k in ['translation', 'size', 'rotation', 'instance_token'] }
            instance_pose_next = { k : ann_next[k] for k in ['translation','size', 'rotation', 'instance_token'] }                       
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
            
            track_forward_non[sd_token][box['instance_token']] = {'translation': trans_sd, 'size':box['size'], 'rotation': angle_sd_q, 'instance_token': box['instance_token'] }
        

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
    
    # In resize function align_corners = false
    im = resize(im, (h_im,w_im,3), order=1, preserve_range=True, anti_aliasing=False) 
    im = im.astype('uint8')
    im = im[y_cutoff:,...]
    return im


def plt_depth_on_im(depth_map, im):
    
    h,w = im.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    msk = depth_map > 0
    
    plt.figure()
    plt.imshow(im) 
    plt.scatter(x_map[msk], y_map[msk], c=depth_map[msk], s=1, cmap='jet')
    plt.colorbar()
    
    
def plt_flow_on_im(flow, im, skip = 3):
    
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


def flow_l2_error(flow_lidar, flow_im):
    msk1 = flow_lidar[:,:,0] != 0
    msk2 = flow_lidar[:,:,1] != 0
    msk = np.logical_or(msk1,msk2)
    
    error = flow_lidar - flow_im
    l2 = (error[...,0] ** 2 + error[...,1] ** 2) ** 0.5
    l2 = l2 * msk
    
    return l2


def filter_occlusion(depth_map, msk_map_low_h, flow_lidar, flow_im, thres=4):
    msk = depth_map > 0
    
    error = flow_lidar - flow_im
    l2 = (error[...,0] ** 2 + error[...,1] ** 2) ** 0.5
    
    msk_occ2 = np.logical_and(msk, l2 > thres)    
    y_list, x_list = np.where(msk_occ2==True)
        
    depth_map[msk_occ2] = 0
    msk_map_low_h[msk_occ2] = False
    flow_lidar[y_list, x_list, :] = [0,0]
    
    return depth_map, msk_map_low_h, flow_lidar
    

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
   
    
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default='d:/Lab/Dataset/nuscenes', help='dataset directory')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='dataset split')
    
    args = parser.parse_args()    
    dir_data = args.dir_data    
    version = args.version
    
   
    nusc = NuScenes(version, dataroot = dir_data, verbose=False)
    
    sample_idx = 250
       
    
    box_tracks_key = {}
    box_tracks_non = {}
    # for i in range(-1, 4+1):
    for i in range(-1, 4+1):
        track_forward_key, track_forward_non = get_tracks_forward(nusc, sample_idx + i)        
        box_tracks_key = update_key_tracks(box_tracks_key, track_forward_key)
        box_tracks_non = update_non_key_tracks(box_tracks_non, track_forward_non)
        
    box_tracks_key.update(box_tracks_non)            
    box_tracks = box_tracks_key
        
    
    n_backward, n_forward = 9, 42
    x1, y1, depth1, x2, y2, depth2, msk_low_h = merge_lidar(nusc, sample_idx, n_backward, n_forward, box_tracks, n_skip=2)
    
    depth_map1, flow, msk_map_low_h = cal_depthMap_flow(x1, y1, depth1, x2, y2, depth2, msk_low_h, downsample_scale=4, y_cutoff=33)
     

    im1, _ = load_im(nusc, sample_idx)
    im1 = downsample_im(im1, downsample_scale=4, y_cutoff=33)
    # im2 = downsample_im(im2, downsample_scale=4, y_cutoff=33)

    
    plt.close('all')
    

    plt.figure()
    plt.imshow(im1)
    
    plt_depth_on_im(depth_map1, im1)   
    plt_flow_on_im(flow, im1, skip=2)
       
    plt.figure()
    plt.imshow(depth_map1, cmap='jet') 
    
    
    out_dir = join(dir_data, 'prepared_data_dense')
    
    im_flow = np.load(join(out_dir, '%05d_flow.npy' % sample_idx))
    plt_flow_on_im(im_flow, im1, skip=5)
    
    
    error = flow_l2_error(flow, im_flow)
    
    thres_occ = 3
    
    plt_depth_on_im(error > thres_occ, im1)

    
    depth_map1, msk_map_low_h, flow = filter_occlusion(depth_map1, msk_map_low_h, flow, im_flow, thres = thres_occ)
    
    plt_depth_on_im(depth_map1, im1)
    
    plt.figure()
    plt.imshow(msk_map_low_h)
    
    plt_depth_on_im(depth_map1 * msk_map_low_h, im1)
    
    
    matrix = np.load(join(out_dir, '%05d_matrix.npz' % sample_idx))
    K = matrix['K']
    
    uv2 = lidarFlow2uv(flow, K, depth_map1, downsample_scale=4, y_cutoff=33)
    
    
    # h, w = depth_map2.shape
    # uv1 = cal_uv1(h, w, K, downsample_scale=4, y_cutoff=33)
    
   
    
  
    
