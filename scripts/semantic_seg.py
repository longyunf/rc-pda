'''
Run command:
python semantic_seg.py  TEST.MODEL_FILE  path/to/panoptic_deeplab_R101_os32_cityscapes.pth

Based on Panoptic-DeepLab (https://github.com/bowenc0221/panoptic-deeplab)

'''

import sys
import os
from os.path import join

deeplab_path = join(os.path.dirname(__file__), '..', 'external', 'panoptic-deeplab', 'tools')
if deeplab_path not in sys.path:
    sys.path.insert(0, deeplab_path)

from functools import reduce
import argparse
import glob
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.backends.cudnn as cudnn

import _init_paths
from segmentation.config import config, update_config
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.model.post_processing import get_semantic_segmentation
import segmentation.data.transforms.transforms as T


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--dir_data', type=str)

    args = parser.parse_args()
        
    this_dir =  os.path.dirname(__file__)
    if args.cfg == None:
        args.cfg = join(this_dir, '..', 'external', 'panoptic-deeplab', 'configs', 'panoptic_deeplab_R101_os32_cityscapes.yaml')
        
    if args.dir_data == None:
        args.dir_data = os.path.join(this_dir, '..', 'data')
        
    update_config(config, args)
    
    return args


def read_image(file_name, format=None):
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image



def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    if len(gpus) > 1:
        raise ValueError('Test only supports single core.')
    device = torch.device('cuda:{}'.format(gpus[0]))

    # build model
    model = build_segmentation_model_from_cfg(config)
    model = model.to(device)
    model_state_file = config.TEST.MODEL_FILE

    if os.path.isfile(model_state_file):
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
        model.load_state_dict(model_weights, strict=True)
    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')
    
    
    out_dir = join(args.dir_data, 'prepared_data')    
    input_list = np.array(np.sort(glob.glob(join(out_dir, '*im.jpg'))))
    

    model.eval()

    # build image demo transform
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                config.DATASET.MEAN,
                config.DATASET.STD
            )
        ]
    )
    
    
    N = len(input_list)
    
    with torch.no_grad():
        for i, fname in enumerate(input_list):
            if isinstance(fname, str):
                # load image
                raw_image = read_image(fname, 'RGB')
            else:
                NotImplementedError("Inference on video is not supported yet.")
            
            # pad image
            raw_shape = raw_image.shape[:2]
            raw_h = raw_shape[0]
            raw_w = raw_shape[1]
            new_h = (raw_h + 31) // 32 * 32 + 1
            new_w = (raw_w + 31) // 32 * 32 + 1
            input_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            input_image[:, :] = config.DATASET.MEAN
            input_image[:raw_h, :raw_w, :] = raw_image

            image, _ = transforms(input_image, None)
            image = image.unsqueeze(0).to(device)

            # network
            out_dict = model(image)
            torch.cuda.synchronize(device)

            # post-processing
            semantic_pred = get_semantic_segmentation(out_dict['semantic'])
                        
            semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
            # crop predictions
            semantic_pred = semantic_pred[:raw_h, :raw_w]            
            # car 13, truck 14, bus 15
            vehicle_seg = reduce(np.logical_or, [semantic_pred==13, semantic_pred==14, semantic_pred==15])                            
                           
            path_seg = fname[:-6] + 'seg.npy' 
            np.save(path_seg, vehicle_seg)            
            print('compute segmentation %d/%d' % ( i, N ) )
            


if __name__ == '__main__':
    main()

