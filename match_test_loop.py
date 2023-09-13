from operator import truediv
from time import time
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd

# wzy add >>>>
import cv2
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch

from utils import (compute_pose_error, compute_epipolar_error,
                            estimate_pose, make_matching_plot,
                            error_colormap, AverageTimer, pose_auc, read_image,
                            rotate_intrinsics, rotate_pose_inplane,
                            scale_intrinsics)

torch.set_grad_enabled(False)
# wzy add <<<<


# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# or DISK+LightGlue
# extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image('/home/zph/projects/LightGlue/assets/10_1683599558_173581.jpg').cuda()
image1 = load_image('/home/zph/projects/LightGlue/assets/10_1683599558_070450.jpg').cuda()

timer = AverageTimer(newline=True)
for i in range(1,20):
    # extract local features
    feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    print('number of points0: ', len(points0))

    # convert CUDA to CPU
    image0_cpu = image0.cpu().numpy().transpose((1, 2, 0))  # CxHxW to HxWxC
    image1_cpu = image1.cpu().numpy().transpose((1, 2, 0))  # CxHxW to HxWxC
    matches = matches.cpu().numpy()
    points0 = points0.cpu().numpy()
    points1 = points1.cpu().numpy()
timer.update('match loop sum')
timer.print()

do_viz = False
fast_viz = False
opencv_display = True
show_keypoints = True
viz_path = '/home/zph/projects/LightGlue/assets/match_test.png'
mconf = np.ones(len(points0))


if do_viz:
    # Visualize the matches.
    color = cm.jet(mconf)
    text = [
        'LightGlue',
        'Keypoints: {}:{}'.format(len(points0), len(points1)),
        'Matches: {}'.format(len(points0)),
    ]

    # Display extra parameter info.
    small_text = [
        'Image Pair: {}:{}'.format('kun0', 'kun1'),
    ]

    make_matching_plot(
        image0, image1, points0, points1, points0, points1, color,
        text, viz_path, show_keypoints,
        fast_viz, opencv_display, 'Matches', small_text)

    timer.update('viz_match')
