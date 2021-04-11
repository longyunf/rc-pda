# ------------------------------------------------------------------------------
# Demo code.
# Example command:
# python tools/demo.py --cfg PATH_TO_CONFIG_FILE \
#   --input-files PATH_TO_INPUT_FILES \
#   --output-dir PATH_TO_OUTPUT_DIR
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------
'''
This is based on Panoptic-DeepLab (code and models are available from https://github.com/bowenc0221/panoptic-deeplab)

run command:
python semantic_seg.py   TEST.MODEL_FILE  ~/Dataset/pre-trained_models/panoptic_deeplab_R101_os32_cityscapes.pth

'''


import sys
dir_panoptic_deeplab = '/home/longyunf/Repos/panoptic-deeplab'
dir_panoptic_deeplab_tools = '/home/longyunf/Repos/panoptic-deeplab/tools'

sys.path.append(dir_panoptic_deeplab)
sys.path.append(dir_panoptic_deeplab_tools)

import matplotlib.pyplot as plt
from os.path import join
from functools import reduce

import argparse
import cv2
import os
import pprint
import logging
import time
import glob

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils import save_debug_images
from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
from segmentation.utils import save_annotation, save_instance_annotation, save_panoptic_annotation
import segmentation.data.transforms.transforms as T
from segmentation.utils import AverageMeter
from segmentation.data import build_test_loader_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default = '/home/longyunf/Repos/panoptic-deeplab/configs/panoptic_deeplab_R101_os32_cityscapes.yaml',
                        type=str)
    parser.add_argument('--input-files',
                        help='input files, could be image, image list or video',
                        default ='/home/longyunf/Dataset/nuscenes_mini/prepared_data_v2/00029_im_in.jpg',
                        type=str)
    parser.add_argument('--extension',
                        help='file extension if input is image list',
                        default='.png',
                        type=str)
    parser.add_argument('--merge-image',
                        help='merge image with predictions',
                        action='store_true')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--dir_data', type=str, default='/home/longyunf/media/nuscenes', help='dataset directory')

    args = parser.parse_args()
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


class CityscapesMeta(object):
    def __init__(self):
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.label_divisor = 1000
        self.ignore_label = 255

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap


def main():
    args = parse_args()

    logger = logging.getLogger('demo')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=args.dir_data, name='demo')

    # logger.info(pprint.pformat(args))
    # logger.info(config)

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

    # Change ASPP image pooling
    # output_stride = 2 ** (5 - sum(config.MODEL.BACKBONE.DILATION))
    # train_crop_h, train_crop_w = config.TEST.CROP_SIZE
    # scale = 1. / output_stride
    # pool_h = int((float(train_crop_h) - 1.0) * scale + 1.0)
    # pool_w = int((float(train_crop_w) - 1.0) * scale + 1.0)

    # model.set_image_pooling((pool_h, pool_w))

    # logger.info("Model:\n{}".format(model))
    model = model.to(device)

    try:
        # build data_loader
        data_loader = build_test_loader_from_cfg(config)
        meta_dataset = data_loader.dataset
        save_intermediate_outputs = True
    except:
        logger.warning(
            "Cannot build data loader, using default meta data. This will disable visualizing intermediate outputs")
        if 'cityscapes' in config.DATASET.DATASET:
            meta_dataset = CityscapesMeta()
        else:
            raise ValueError("Unsupported dataset: {}".format(config.DATASET.DATASET))
        save_intermediate_outputs = False

    # load model
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

    if os.path.isfile(model_state_file):
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
            logger.info('Evaluating a intermediate checkpoint.')
        model.load_state_dict(model_weights, strict=True)
        logger.info('Test model loaded from {}'.format(model_state_file))
    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')
    
    
    dir_data = args.dir_data
    out_dir = join(dir_data, 'prepared_data_dense')
    
    input_list = np.array(np.sort(glob.glob(join(out_dir, '*im.jpg'))))
    

    # # load images
    # input_list = []
    # if os.path.exists(args.input_files):
    #     if os.path.isfile(args.input_files):
    #         # inference on a single file, extract extension
    #         ext = os.path.splitext(os.path.basename(args.input_files))[1]
    #         if ext in ['.png', '.jpg', '.jpeg']:
    #             # image file
    #             input_list.append(args.input_files)
    #         elif ext in ['.mpeg']:
    #             # video file
    #             # TODO: decode video and convert to image list
    #             raise NotImplementedError("Inference on video is not supported yet.")
    #         else:
    #             raise ValueError("Unsupported extension: {}.".format(ext))
    #     else:
    #         # inference on a directory
    #         for fname in glob.glob(os.path.join(args.input_files, '*' + args.extension)):
    #             input_list.append(fname)
    # else:
    #     raise ValueError('Input file or directory does not exists: {}'.format(args.input_files))




    # Test loop
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
    
    net_time = AverageMeter()
    post_time = AverageMeter()
    try:
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
                start_time = time.time()
                out_dict = model(image)
                torch.cuda.synchronize(device)
                net_time.update(time.time() - start_time)

                # post-processing
                start_time = time.time()
                semantic_pred = get_semantic_segmentation(out_dict['semantic'])

                
                # save predictions
                semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                # panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()

                # crop predictions
                semantic_pred = semantic_pred[:raw_h, :raw_w]
                
                # car 13, truck 14, bus 15
                vehicle_seg = reduce(np.logical_or, [semantic_pred==13, semantic_pred==14, semantic_pred==15])                
                # person 11, rider 12, motorcycle 17, bicycle 18
                # person_seg = reduce(np.logical_or, [semantic_pred==11, semantic_pred==12, semantic_pred==17, semantic_pred==18])
                
                
                # seg = np.stack([vehicle_seg, person_seg], axis=2)
                seg = vehicle_seg
                               
                path_seg = fname[:-6] + 'seg.npy'           
                np.save(path_seg, seg)
                
                print('compute segmentation %d/%d' % ( i, N ) )
                
                
                
                # plt.figure()
                # plt.imshow(raw_image)
                # plt.show()
                
                # plt.figure()
                # plt.imshow(semantic_pred)
                # plt.show()
                
                # plt.figure()
                # plt.imshow(vehicle_seg)
                # plt.show()
                

 
    except Exception:
        logger.exception("Exception during demo:")
        raise
    # finally:
    #     logger.info("Demo finished.")
    #     if save_intermediate_outputs:
    #         logger.info("Intermediate outputs saved to {}".format(raw_out_dir))
    #     logger.info("Semantic predictions saved to {}".format(semantic_out_dir))
    #     logger.info("Instance predictions saved to {}".format(instance_out_dir))
    #     logger.info("Panoptic predictions saved to {}".format(panoptic_out_dir))


if __name__ == '__main__':
    main()

