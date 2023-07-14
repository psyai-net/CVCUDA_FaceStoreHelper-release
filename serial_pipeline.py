import os
import time

import config as cfg
from api import run_inpainting_1, run_inpainting_2, run_inpainting_3, run_postprocessing

import os
import cv2
import numpy as np
import imageio
import traceback
import torch
import argparse
import pickle
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip

import config as cfg
from config import log, params

# from Inference_inpainting_tools import set_realesrgan

from torchvision.transforms.functional import normalize
from torch.utils.data import Dataset, DataLoader

from basicsr.utils import img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY
import cvcuda
import nvcv
import numexpr as ne
import numba as nb
import cvcuda_utils



# def gen_video(audio_path, video_source_path, resource, wav2lip_model, affine_matrices, inverse_affine_matrices, codeformer_net, model_face_parser):


#     cropped_face_all, cvcuda_inverse_affine_matrices = run_inpainting_1(wav2lip_result, affine_matrices, inverse_affine_matrices)
#     print("block2--codeformer1耗时：", time.time() - start_time)
#     start_time = time.time()

#     out, restored_face = run_inpainting_2(cropped_face_all, wav2lip_result, codeformer_net, model_face_parser)
#     print("block3--codeformer2耗时：", time.time() - start_time)
#     start_time = time.time()

#     cfg.output_imglist = run_inpainting_3(out, wav2lip_result, restored_face, cvcuda_inverse_affine_matrices)
#     print("block4--codeformer3耗时：", time.time() - start_time)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./testdata/video_based/123.mp4', 
            help='Input image, video or folder. Default: ./testdata/video_based/123.mp4')
    
    parser.add_argument('-o', '--output_path', type=str, default="./results/result.mp4", 
            help='Output folder. Default: ./results/result.mp4')

    parser.add_argument('-a', '--audio_path', type=str, default=None, 
        help='Output folder. Default: origin audio')

    args = parser.parse_args()
    input_video = False
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_img_{w}'
    elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter
        input_img_list = []
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps() # 帧率
        video_name = os.path.basename(args.input_path)[:-4]
        # result_root = f'results/{video_name}_{w}'
        input_video = True
        vidreader.close()
    else: # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        # result_root = f'results/{os.path.basename(args.input_path)}_{w}'

        # 通过本地资源包读取人脸正脸化需要的数据
    with open(params['app']['codeformer_source_path'] + 'all_landmarks_5.pkl', 'rb') as f1:
        all_landmarks_5 = pickle.load(f1)
    with open(params['app']['codeformer_source_path'] + 'det_faces.pkl', 'rb') as f2:
        det_faces = pickle.load(f2)
    with open(params['app']['codeformer_source_path'] + 'affine_matrices.pkl', 'rb') as f3:
        affine_matrices = pickle.load(f3)
    with open(params['app']['codeformer_source_path'] + 'inverse_affine_matrices.pkl', 'rb') as f4:
        inverse_affine_matrices = pickle.load(f4)
        
    start_time_all = time.time()
    start_time = time.time()
    cropped_face_all, cvcuda_inverse_affine_matrices = run_inpainting_1(input_img_list, affine_matrices, inverse_affine_matrices)
    print("block1--codeformer1耗时：", time.time() - start_time)
    start_time = time.time()

    out, restored_face = run_inpainting_2(cropped_face_all)
    print("block2--codeformer2耗时：", time.time() - start_time)
    start_time = time.time()

    cfg.output_imglist = run_inpainting_3(out, input_img_list, restored_face, cvcuda_inverse_affine_matrices)
    print("block3--codeformer3耗时：", time.time() - start_time)
    
    save_video_path = "./outlip.mp4"
    start_time = time.time()
    save_video_final_path = run_postprocessing(cfg.output_imglist, fps, save_video_path, args.input_path)
    print("后处理阶段耗时：", time.time() - start_time)
    print("使用codeformer处理整个视频总耗时：", time.time() - start_time_all)