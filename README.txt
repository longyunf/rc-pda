1. Data preparation
1) split_trainval_small3.py      # split data
2) prepare_flow_im.py            # extract images for flow computation
3) compute_flow12.py            # compute image flow from im1 to im2
4) cal_cam_matrix.py              # compute camera intrinsic matrix and transformation from cam1 to cam2
5) cal_im_flow2uv.py               # transform image flow to normalized expression (u2,v2)
6) semantic_seg.py                  # compute vehicle semantic segmentation
7) cal_dense_gt3.py                # compute dense ground truth (depth1, u2, v2) and low height mask
8) cal_single_gt.py                   # compute single frame gt (depth, u2, v2)
9) cal_short_radar.py               # compute merged radar (5 frames)
10) cal_single_radar.py            # single frame radar
11) gen_h5_file3.py                 # create .h5 dataset file

2. Estimate radar camera assoication
train_aff.py         # train
test_aff.py          # test

3. Generate enhanced radar depth (RC-PDA)
expand_multi_radar.py

4. Train depth completion by using the enhanced depth
train_rd_ft4.py         # train
test_rd_ft4.py          # test

