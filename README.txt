1. Data preparation
1) split_trainval.py          # split data
2) prepare_flow_im.py         # extract images for flow computation
3) cal_flow.py                # compute image flow from im1 to im2
4) cal_cam_matrix.py          # compute camera intrinsic matrix and transformation from cam1 to cam2
5) cal_im_flow2uv.py          # transform image flow to normalized expression (u2,v2)
6) semantic_seg.py            # compute vehicle semantic segmentation
7) cal_gt.py                  # compute dense ground truth (depth1, u2, v2) and low height mask
8) cal_radar.py               # compute merged radar (5 frames)
9) gen_h5_file3.py            # create .h5 dataset file

2. Estimate radar camera assoication
train_pda.py                  # train
test_pda.py                   # test

3. Generate enhanced radar depth (RC-PDA)
cal_mer.py

4. Train depth completion by using the enhanced depth
train_depth.py                # train
test_depth.py                 # test


