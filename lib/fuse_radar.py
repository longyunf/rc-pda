from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from functools import reduce
from pyquaternion import Quaternion
import skimage.io as io
import os
from os.path import join
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def merge_selected_radar(nusc, sample_idx, frame_range=[0,12]):
    '''
    frame_range: [a,b]; (b-a+1) frames; 0 represents current frame
    
    '''
    
    def cal_matrix_refCam_from_global(cam_data):
        ref_pose_rec = nusc.get('ego_pose', cam_data['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])    
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)        
        M_ref_from_global = reduce(np.dot, [ref_from_car, car_from_global])
        
        return M_ref_from_global
    
    def current_2_ref(current_sd_rec, M_refCam_from_global, ref_time, min_distance=1):        
        current_pc = pc.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        trans_matrix = reduce(np.dot, [M_refCam_from_global, global_from_car, car_from_current])
        
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  
        
        vx_comp, vy_comp = current_pc.points[8], current_pc.points[9]
        
        # Use Doppler velocity to compensate object motion        
        current_pc.points[0] += vx_comp * time_lag
        current_pc.points[1] += vy_comp * time_lag
        
        current_pc.transform(trans_matrix)
        
        time_lag = abs(time_lag)
        
        return current_pc, time_lag
    
    def proj2im(pc_cam, cam_data, min_z = 2): 
        
        cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])         
        depth = pc_cam.points[2]
        RCSs = pc_cam.points[5]
        
        vx_comp, vy_comp = pc_cam.points[8], pc_cam.points[9]
        v_comp = (vx_comp ** 2 + vy_comp ** 2) ** 0.5
        
        
        msk = pc_cam.points[2] >= min_z       
        points = view_points(pc_cam.points[:3, :], np.array(cs_rec['camera_intrinsic']), normalize=True)        
        x, y = points[0], points[1]
        
        
        msk =  reduce(np.logical_and, [x>0, x<1600, y>0, y<900, msk])        
        return x, y, depth, RCSs, v_comp, msk
    

    sample_rec = nusc.sample[sample_idx]
    radar_token = sample_rec['data']['RADAR_FRONT']        
    radar_sample = nusc.get('sample_data', radar_token)
    
    radar_token = radar_sample['next']
    radar_sample = nusc.get('sample_data', radar_token)     # make next radar frame the latest frame

        
    pcl_path = os.path.join(nusc.dataroot, radar_sample['filename'])
    RadarPointCloud.disable_filters()
    pc = RadarPointCloud.from_file(pcl_path)   
    
    all_pc1 = RadarPointCloud(np.zeros((pc.nbr_dims(), 0)))
    all_pc2 = RadarPointCloud(np.zeros((pc.nbr_dims(), 0)))
    
    all_times1 = np.zeros((0,))
    all_times2 = np.zeros((0,))
         
    # Get reference pose and timestamp.    
    cam_token = sample_rec['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)   
    cam_token1 = cam_data['next']                    
    cam_data1 = nusc.get('sample_data', cam_token1)
    cam_token2 = cam_data1['next'] 
    cam_data2 = nusc.get('sample_data', cam_token2)
    
    M_refCam1_from_global = cal_matrix_refCam_from_global(cam_data1)
    M_refCam2_from_global = cal_matrix_refCam_from_global(cam_data2)
    
    ref_time1 = 1e-6 * cam_data1['timestamp']
    ref_time2 = 1e-6 * cam_data2['timestamp']
    

    # Aggregate current and previous sweeps.
    current_sd_rec = radar_sample        
    k=0    
    while k < frame_range[0]:
        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            return None, None
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])
        k += 1
        

    nsweeps = frame_range[1] - frame_range[0] + 1
        
    for _ in range(nsweeps):
                
        current_pc1, time_lag1 = current_2_ref(current_sd_rec, M_refCam1_from_global, ref_time1)
        current_pc2, time_lag2 = current_2_ref(current_sd_rec, M_refCam2_from_global, ref_time2)
              
        times1 = time_lag1 * np.ones((current_pc1.nbr_points(),))
        times2 = time_lag2 * np.ones((current_pc2.nbr_points(),))

        all_times1 = np.hstack((all_times1, times1))
        all_times2 = np.hstack((all_times2, times2))

        all_pc1.points = np.hstack((all_pc1.points, current_pc1.points))
        all_pc2.points = np.hstack((all_pc2.points, current_pc2.points))

        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])
            
            
    x1, y1, depth1, rcs1, v_comp1, msk1 = proj2im(all_pc1, cam_data1) 
    x2, y2, depth2, rcs2, v_comp2, msk2 = proj2im(all_pc2, cam_data2)
        
    rcs = rcs2
    v_comp = v_comp2       
    msk = msk1 * msk2
        
    x1, y1, x2, y2, depth1, depth2, all_times1, all_times2, rcs, v_comp = x1[msk], y1[msk], x2[msk], y2[msk], depth1[msk], depth2[msk], all_times1[msk], all_times2[msk], rcs[msk], v_comp[msk]
                  
    return x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp


def cal_depthMap_flow(x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp, downsample_scale, y_cutoff):
    
    h_im, w_im = 900, 1600
    h_new = int( h_im / downsample_scale )
    w_new = int( w_im / downsample_scale ) 
      
    depth_map1 = np.zeros( (h_new, w_new) , dtype=float)
    time_map1 = np.zeros( (h_new, w_new) , dtype=float)
    rcs_map1 = np.zeros( (h_new, w_new) , dtype=float)
    v_comp_map1 = np.zeros( (h_new, w_new) , dtype=float)
        
    flow = np.zeros( (h_new, w_new, 2) , dtype=float)      # dx, dy
     
    x1 = (x1 + 0.5) / downsample_scale - 0.5
    y1 = (y1 + 0.5) / downsample_scale - 0.5
    x2 = (x2 + 0.5) / downsample_scale - 0.5
    y2 = (y2 + 0.5) / downsample_scale - 0.5
        
    x1 = np.clip(x1, 0, w_new - 1)
    x2 = np.clip(x2, 0, w_new - 1)
    y1 = np.clip(y1, 0, h_new - 1)
    y2 = np.clip(y2, 0, h_new - 1)
           
    offset_rcs = 10
    rcs += offset_rcs
    
    for i in range(len(x1)):
        x1_one, y1_one = int(round( x1[i] )), int(round( y1[i] ))
                
        if depth_map1[y1_one,x1_one] == 0:
            depth_map1[y1_one,x1_one] = depth1[i]        
            flow[y1_one,x1_one, ...] = [x2[i] - x1[i], y2[i] - y1[i]] 
            time_map1[y1_one,x1_one] = all_times1[i]
            rcs_map1[y1_one,x1_one] = rcs[i]
            v_comp_map1[y1_one,x1_one] = v_comp[i]
                          
        elif depth_map1[y1_one,x1_one] > depth1[i]: 
            depth_map1[y1_one,x1_one] = depth1[i]
            flow[y1_one,x1_one, ...] = [x2[i] - x1[i], y2[i] - y1[i]] 
            time_map1[y1_one,x1_one] = all_times1[i]
            rcs_map1[y1_one,x1_one] = rcs[i]
            v_comp_map1[y1_one,x1_one] = v_comp[i]
              
    depth_map1, flow = depth_map1[y_cutoff:,...], flow[y_cutoff:,...]
    time_map1, rcs_map1, v_comp_map1 = time_map1[y_cutoff:,...], rcs_map1[y_cutoff:,...], v_comp_map1[y_cutoff:,...]
    
    v_comp_map1 = (v_comp_map1 > 0.5).astype(float)
           
    return depth_map1, flow, time_map1, rcs_map1, v_comp_map1


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


def flow_l2_error(flow_radar, flow_im):
    msk1 = flow_radar[:,:,0] != 0
    msk2 = flow_radar[:,:,1] != 0
    msk = np.logical_or(msk1,msk2)
    
    error = flow_radar - flow_im
    l2 = (error[...,0] ** 2 + error[...,1] ** 2) ** 0.5
    l2 = l2 * msk
    
    return l2


def radarFlow2uv(flow, K, depth_map, downsample_scale=4, y_cutoff=33):
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


    
    
    