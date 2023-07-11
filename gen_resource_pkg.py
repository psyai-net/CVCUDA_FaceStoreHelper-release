
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import pickle
import sys

from config import log, params


import subprocess

import config as cfg
from Inference_wav2lip_tools import  get_audio_frame_list, load_model, Wav2lip_Dataset
from config import log, params
from facelib.utils.face_restoration_helper import FaceRestoreHelper

from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from basicsr.utils.registry import ARCH_REGISTRY

'''
    生成资源包
    生成之后直接在params['app']['video_based_path']目录下，每个虚拟人ID一个子目录，子目录里放该人的所有资源包文件
'''
    
# 直接跑codeformer的保存pkl模块



'''
    获取视频帧序列
'''
def get_video_frame_list(face_video_path):
    video_stream = cv2.VideoCapture(face_video_path)
    fps = int(video_stream.get(cv2.CAP_PROP_FPS))

    log.logger.info('Reading video frames...')

    full_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break

        full_frames.append(frame)
    return full_frames, fps



# 跑 codeformer 的 pkl 转 成 wav2lip认识的pkl

def face_detect_by_codeformer(images, codeformer_det_faces_pkl_path ,video_name, fps, save_pkl_path):
    
    log.logger.info('把codeformer格式的pkl转为wav2lip能接受的')
    
    '''
    输入 images 把codeformer输出的pkl转为wav2lip可接受的pkl，使用localpose时可直接读此pkl
    '''
    with open(codeformer_det_faces_pkl_path, 'rb') as f:
        face_data = pickle.load(f)
        
    
    results = []
    pady1, pady2, padx1, padx2 = params['app']['wav2lip']['pads']
    for image, bbox in zip(images, face_data):
        # import pdb;pdb.set_trace()
        
        y1 = round( max(0, bbox[0][1] - pady1) )
        y2 = round( min(image.shape[0], bbox[0][3] + pady2) )
        x1 = round( max(0, bbox[0][0] - padx1) )
        x2 = round( min(image.shape[1], bbox[0][2] + padx2) )
        
        results.append([x1, y1, x2, y2])
    
    boxes = np.array(results)
    
    # import pdb;pdb.set_trace()
    
    
    if not params['app']['wav2lip']['nosmooth']: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    
    # print("datagen:",results)  ## 把这个results写死
    # import pdb;pdb.set_trace()
    
    # print("将转成wav2lip格式的list保存到文件")
    wav2lip_det_faces_pkl_path = os.path.dirname(codeformer_det_faces_pkl_path)
    # print("wav2lip_det_faces_pkl_path:", wav2lip_det_faces_pkl_path)
    
    # import pdb
    # pdb.set_trace()
    
    with open(os.path.join(save_pkl_path, "%s.pkl" % video_name), "wb") as f5:
        pickle.dump(results, f5)
    
    # import pdb;pdb.set_trace()
        
    return None

'''
    获取平滑之后的boxes
'''
def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def gen_resource_pkg(video_source_path, output_path):
    
    video_name, file_ext = os.path.splitext(os.path.basename(video_source_path))
    
    save_pkl_path = os.path.join(output_path, str(video_name))
    
    os.makedirs(save_pkl_path, exist_ok=True)
    
    imglist, fps = get_video_frame_list(video_source_path)
    
    if fps == 0:
        log.logger.error(video_source_path,"导出的 1080 1920 视频的 有问题！！！")
        sys.exit(0)
    
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                            connect_list=['32', '64', '128', '256']).to(cfg.device)

    checkpoint = torch.load("./checkpoint/codeformer.pth")['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()      
    
    face_helper = FaceRestoreHelper(
        params['app']['inpainting']['upscale'],
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext='png',
        use_parse=True,
        device=cfg.device)  # 人脸关键点检测用的是 retinaface_resnet50
    
    
    total_all_landmarks_5 = []
    total_det_faces = []
    total_affine_matrices = []
    total_inverse_affine_matrices = []
    
    
    output_imglist = []
    
    for frame_idx in tqdm(range(len(imglist)), desc='正在逐帧处理视频，获取资源包'):
        # print("frame_idx:", frame_idx)
        
              
        bg_upsampler = None
        face_upsampler = None
        
        
        img = imglist[frame_idx]
        face_helper.clean_all()
        face_helper.read_image(img)

        # import pdb;pdb.set_trace()
        num_det_faces = face_helper.get_face_landmarks_5(only_center_face=params['app']['inpainting']['only_center_face'], resize=640, eye_dist_threshold=5)


        # align and warp each face
        # import pdb;pdb.set_trace()
        face_helper.align_warp_face()  # 对齐

        # face restoration for each cropped face
        # print("face_helper.cropped_faces:", face_helper.cropped_faces)
        
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(cfg.device)

            # print("cropped_face_t.shape",cropped_face_t.shape)
            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=params['app']['inpainting']['fidelity_weight'], adain=True)[0]    # [1, 3, 512, 512]
                    output_copy = output.clone()

                    restored_face = tensor2img(output_copy, rgb2bgr=True, min_max=(-1, 1))  # restored_face [512, 512, 3]

                    normalize(output, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    with torch.no_grad():
                        out = face_helper.face_parse(output)[0]

                    out = out.argmax(dim=1).squeeze().cpu().numpy()  
                      
                    # print("out.shape",out.shape)

                del output
                torch.cuda.empty_cache()
            except Exception as e:
                log.logger.error(f'\tFailed inference for CodeFormer: %s' % (traceback.format_exc()))
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if params['app']['inpainting']['has_aligned'] is False:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=params['app']['inpainting']['upscale'])[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            # import pdb;pdb.set_trace()
            if params['app']['inpainting']['face_upsample'] and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(face_parse_out=out, upsample_img=bg_img, draw_box=params['app']['inpainting']['draw_box'],
                                                                        face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(face_parse_out=out, upsample_img=bg_img, draw_box=params['app']['inpainting']['draw_box'])
            
            output_imglist.append(restored_img)
            
        total_all_landmarks_5.append(face_helper.all_landmarks_5)
        total_det_faces.append(face_helper.det_faces)
        total_affine_matrices.append(face_helper.affine_matrices)
        total_inverse_affine_matrices.append(face_helper.inverse_affine_matrices)
            

    with open(os.path.join(save_pkl_path, "all_landmarks_5.pkl"), 'wb') as f1:
        pickle.dump(total_all_landmarks_5, f1) 
    with open(os.path.join(save_pkl_path, "det_faces.pkl"), 'wb') as f2:
        pickle.dump(total_det_faces, f2)        
    with open(os.path.join(save_pkl_path, "affine_matrices.pkl"), 'wb') as f3:
        pickle.dump(total_affine_matrices, f3)     
    with open(os.path.join(save_pkl_path, "inverse_affine_matrices.pkl"), 'wb') as f4:
        pickle.dump(total_inverse_affine_matrices, f4)      
        
    log.logger.info("所有资源包 pickle dump 完成")
    
    
def change_w_h_25fps(origin_video_path, resize_video_path, fps=25, width=1080, height=1920):

    directory = origin_video_path
    output_path = resize_video_path

    for filename in tqdm(os.listdir(directory), desc='正在处理ffmpeg resize图片'):
        if filename.endswith(".mp4"):
            input_path_tmp = os.path.join(directory, filename)
            output_path_tmp = os.path.join(output_path, filename)
            subprocess.call(f"ffmpeg -i '{input_path_tmp}' -s '{width}'x'{height}' -r '{fps}' '{output_path_tmp}' -loglevel quiet", 
                            shell=True)
        else:
            log.logger.info("文件夹中有非mp4文件: %s，已跳过" % filename)
    

if __name__ == "__main__":
    
    origin_video_path = "./testdata/video_based/" 
    use_resize = False # 是否调整分辨率和帧数  

    #############
    ## 进行调整分辨率和帧数的操作
    ############## 

    # 1-1、调整分辨率及FPS
    if (use_resize==True):
        resize_video_path = "./testdata/resize/"   
        os.makedirs(resize_video_path, exist_ok=True)
        log.logger.info("开始调整分辨率和帧数")
        change_w_h_25fps(origin_video_path, resize_video_path, fps=25)
        log.logger.info("分辨率和帧数调整完成")
        
        file_names = os.listdir(resize_video_path)
        output_path = resize_video_path
        ready_video_path = resize_video_path
        
        
    else:
    # 1-2、不调整分辨率及FPS
        log.logger.info("不调整分辨率")
        file_names = os.listdir(origin_video_path)
        output_path = origin_video_path
        ready_video_path = origin_video_path

    #############
    ## 2、进行保存资源包的操作
    ##############

    # print("file_names",file_names)

    for file_name in tqdm(file_names, desc='正在处理'+ ready_video_path):
        if file_name.endswith('.mp4'):
            gen_resource_pkg(os.path.join(ready_video_path, file_name), output_path)
            
    print("资源包pkl保存完成！！")