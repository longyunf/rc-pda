# Radar-Camera Pixel Depth Association for Depth Completion

![example figure](./images/example.png)
**Example of radar-camera depth completion: (a) raw radar depth, (b) enhanced radar depth, (c) final predicted depth.**


## Directories

rc-pda/
    ├── data/                           							  (This can be a soft link)
                └── nuscenes/                 		           (Download official [nuScenes dataset](https://www.nuscenes.org/))
                             ├── annotations/
                             ├── maps/
                             ├── samples/
                             ├── sweeps/
                             └── v1.0-trainval/
    ├── lib/
    ├── scripts/
    ├── external/                   				   			   (External repositories)
                 ├── panoptic-deeplab/       		        (Clone [Panoptic-DeepLab](https://github.com/bowenc0221/panoptic-deeplab))
                 └── RAFT/                   			         	  (Clone [RAFT](https://github.com/princeton-vl/RAFT))



## Setup
- Create a conda environment called pda
```bash
conda create -n pda python=3.6
```
- Install required packages
```bash
pip install -r requirements.txt
```
- Create folder data/ and download nuScenes dataset
- Create folder external/ and clone external repos

## Code
**1. Data preparation**

```bash
cd scripts

# 1) split data
python split_trainval.py

# 2) extract images for flow computation
python prepare_flow_im.py

# 3) compute image flow from im1 to im2
python cal_flow.py 

# 4) compute camera intrinsic matrix and transformation from cam1 to cam2
python cal_cam_matrix.py 

# 5) transform image flow to normalized expression (u2,v2)
python cal_im_flow2uv.py  

# 6) compute vehicle semantic segmentation
python semantic_seg.py 

# 7) compute dense ground truth (depth1, u2, v2) and low height mask
python cal_gt.py  

# 8) compute merged radar (5 frames)
python cal_radar.py       

# 9) create .h5 dataset file
python gen_h5_file3.py           
```

**2. Estimate radar camera association**
```bash
python train_pda.py        # train
python test_pda.py         # test
```

**3. Generate enhanced radar depth (RC-PDA)**
```bash
python cal_mer.py
```

**4. Train depth completion by using the enhanced depth**

```bash
python train_depth.py        # train
python test_depth.py         # test
```

## Citation
@inproceedings{long2021radar,
  title={Radar-Camera Pixel Depth Association for Depth Completion},
  author={Long, Yunfei and Morris, Daniel and Liu, Xiaoming and Castro, Marcos and Chakravarty, Punarjay and Narayanan, Praveen},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}



