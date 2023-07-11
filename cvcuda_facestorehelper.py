import os
import cv2
import numpy as np
import imageio
import traceback
import torch
import pickle
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip

import config as cfg
from config import log, params
from Inference_wav2lip_tools import get_video_frame_list, get_audio_frame_list, load_model, Wav2lip_Dataset
from Inference_inpainting_tools import set_realesrgan

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


        


def run_inpainting(imglist):
    # bg_upsampler = set_realesrgan()
    
    ### 以下代码替换成任笑田加载模型的方式，正脸化不从资源包读取，而是保留实时生成的方式
    bg_upsampler = None
    face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                            connect_list=['32', '64', '128', '256']).to(cfg.device)

    checkpoint = torch.load(params['app']['inpainting']['codeformer_checkpoint_path'])['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    face_helper = FaceRestoreHelper(
        params['app']['inpainting']['upscale'],
        face_size=512,
        crop_ratio=(1, 1),
        det_model=params['app']['inpainting']['detection_model'],
        save_ext='png',
        use_parse=True,
        device=cfg.device)  # 人脸关键点检测用的是 retinaface_resnet50

    output_imglist = []
    
    
    ### 以上代码替换成任笑田加载模型的方式，正脸化不从资源包读取，而是保留实时生成的方式
    
    
    # 通过本地资源包读取人脸正脸化需要的数据
    with open(params['app']['codeformer_source_path'] + 'all_landmarks_5.pkl', 'rb') as f1:
        all_landmarks_5 = pickle.load(f1)
    with open(params['app']['codeformer_source_path'] + 'det_faces.pkl', 'rb') as f2:
        det_faces = pickle.load(f2)
    with open(params['app']['codeformer_source_path'] + 'affine_matrices.pkl', 'rb') as f3:
        affine_matrices = pickle.load(f3)
    with open(params['app']['codeformer_source_path'] + 'inverse_affine_matrices.pkl', 'rb') as f4:
        inverse_affine_matrices = pickle.load(f4)
    
    #################################################################################
    import time
    aaa = time.time()
    affine_matrices = [item[0].tolist() for item in affine_matrices]
    affine_matrices = np.array(affine_matrices, dtype=np.float32)[:len(imglist)].reshape(-1,6)
    
    inverse_affine_matrices = [item[0].tolist() for item in inverse_affine_matrices]
    inverse_affine_matrices = np.array(inverse_affine_matrices, dtype=np.float32)[:len(imglist)].reshape(-1,6)
    
    cvcuda_affine_matrices = cvcuda_utils.to_nvcv_tensor(affine_matrices, "NC")
    image_batch = nvcv.ImageBatchVarShape(len(imglist))

    # 0.44s
    for index in range(len(imglist)):
        input_nvcv_image = cvcuda_utils.to_nvcv_image( imglist[index] )
        image_batch.pushback(input_nvcv_image)
    out = cvcuda.warp_affine(
        image_batch, cvcuda_affine_matrices, cvcuda.Interp.LINEAR, border_mode=cvcuda.Border.CONSTANT, border_value=[0]
    ) 
    # 
    
    cropped_face_t_list = []
    # bbb = time.time() # 1.5s
    for bimg in out:
        test_input_nvcv_tensor = nvcv.as_tensor(bimg)
        test_input_nvcv_tensor = torch.as_tensor(test_input_nvcv_tensor.cuda(), device="cuda")
        immm = test_input_nvcv_tensor[:,:512,:512,:].squeeze().cpu().numpy()
        cropped_face_t = img2tensor(immm / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t_list.append(cropped_face_t)
    cropped_face_t = torch.stack(cropped_face_t_list)[:2].to(cfg.device)
    # print(time.time() - bbb)
    
    ################### 第二个block
    bbb = time.time()
    with torch.no_grad():
        output = net(cropped_face_t, w=params['app']['inpainting']['fidelity_weight'], adain=True)[0]    # [1, 3, 512, 512]
        output_copy = output.clone()    # torch.Size([12, 3, 512, 512])
        # tensor 2 img
        min_max=(-1, 1)
        output_copy = output_copy.float().detach().cpu().clamp_(*min_max)
        output_copy = (output_copy - min_max[0]) / (min_max[1] - min_max[0])
        output_copy = output_copy.numpy().transpose(0, 2, 3, 1)
        # output_copy = cv2.cvtColor(output_copy, cv2.COLOR_RGB2BGR)
        output_copy_orl = output_copy.copy()
        output_copy[:,:,:,0] = output_copy_orl[:,:,:,2]
        output_copy[:,:,:,2] = output_copy_orl[:,:,:,0]
        restored_face = (output_copy * 255.0).round().astype(np.uint8)
        normalize(output, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        with torch.no_grad():
            out = face_helper.face_parse(output)[0] # torch.Size([12, 19, 512, 512])
            out = out.argmax(dim=1).squeeze().cpu().numpy() # (12, 512, 512)
    del output
    torch.cuda.empty_cache()
    
    ################### 第三个block
    ### 经过codeformer和parse两个网络结束，转成cvcuda格式继续后面的cv2加速
    bbb = time.time()
    h, w, _ = imglist[0].shape
    cvcuda_inverse_affine_matrices = cvcuda_utils.to_nvcv_tensor(inverse_affine_matrices, "NC")
    restore_image_batch = nvcv.ImageBatchVarShape(restored_face.shape[0])
    restored_face = np.ascontiguousarray(restored_face)
    for index in range(restored_face.shape[0]):
        input_nvcv_image = cvcuda_utils.to_nvcv_image(restored_face[index])
        restore_image_batch.pushback(input_nvcv_image)
    tmpsize = cvcuda_utils.clone_image_batch(restore_image_batch, newsize=(w,h)) 
    stream = cvcuda.Stream()
    inv_restored = cvcuda.warp_affine_into(src=restore_image_batch,dst=tmpsize,xform=cvcuda_inverse_affine_matrices,flags=cvcuda.Interp.LINEAR,
        border_mode=cvcuda.Border.CONSTANT,border_value=[0],stream=stream)
    
    for bimg in inv_restored:
        test_input_nvcv_tensor = nvcv.as_tensor(bimg)
        test_input_nvcv_tensor = torch.as_tensor(
                test_input_nvcv_tensor.cuda(), device="cuda"
        )
        img_numpy = test_input_nvcv_tensor.cpu().numpy()
        cv2.imwrite("./inv_restored.png",img_numpy.squeeze())
        break
    
    mask = np.ones((restored_face.shape[0], out.shape[-1], out.shape[-1]), dtype=np.float32)
    mask_image_batch = nvcv.ImageBatchVarShape(restored_face.shape[0])
    for index in range(restored_face.shape[0]):
        input_nvcv_image = cvcuda_utils.to_nvcv_image(mask[index])
        mask_image_batch.pushback(input_nvcv_image)
    tmpsize = cvcuda_utils.clone_image_batch(mask_image_batch, newsize=(w,h)) 
    stream = cvcuda.Stream()
    inv_mask = cvcuda.warp_affine_into(src=mask_image_batch,dst=tmpsize,xform=cvcuda_inverse_affine_matrices,flags=cvcuda.Interp.LINEAR,
        border_mode=cvcuda.Border.CONSTANT,border_value=[0],stream=stream)
    print(time.time() - bbb,'???????????')  # 1 张图片 0.017s
    # for bimg in inv_mask:
    #     test_input_nvcv_tensor = nvcv.as_tensor(bimg)
    erosionmask = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0],2), np.int32) * 2).cuda())
    anchor = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0],2), np.int32) * -1).cuda())
    inv_mask_erosion = cvcuda.morphology(inv_mask, cvcuda.MorphologyType.ERODE,erosionmask,anchor, border=cvcuda.Border.CONSTANT)  

    erosionmask = np.ones((restored_face.shape[0],2), np.int32)
    idx = 0
    # import pdb;pdb.set_trace()
    pasted_face_list = []
    for inv_mask, inv_restore_img in zip(inv_mask_erosion, inv_restored):
        tmpp_inv_mask = torch.as_tensor(nvcv.as_tensor(inv_mask).cuda()).squeeze()[:, :, None]
        tmpp_inv_restore_img = torch.as_tensor(nvcv.as_tensor(inv_restore_img).cuda()).squeeze()
        pasted_face = tmpp_inv_mask * tmpp_inv_restore_img
        pasted_face_list.append(pasted_face)
        total_face_area = np.sum(tmpp_inv_mask.cpu().numpy())
        erosion_radius = (int(total_face_area**0.5) // 20) * 2
        erosionmask[idx] *= erosion_radius
        idx += 1
    
    erosionmask_1 = nvcv.as_tensor(torch.from_numpy(erosionmask).cuda())
    inv_mask_center = cvcuda.morphology(inv_mask_erosion, cvcuda.MorphologyType.ERODE,erosionmask_1,anchor, border=cvcuda.Border.CONSTANT)  
    # import pdb;pdb.set_trace()
    # for bimg in inv_mask_center:
    #     test_input_nvcv_tensor = nvcv.as_tensor(bimg)
    #     test_input_nvcv_tensor = torch.as_tensor(
    #             test_input_nvcv_tensor.cuda(), device="cuda"
    #     )
    #     img_numpy = test_input_nvcv_tensor.cpu().numpy()
    #     cv2.imwrite("./inv_mask_center.png",img_numpy.squeeze()*255)
    #     break
    
    max_ks = np.max(erosionmask) + 1
    kernel_size = nvcv.as_tensor(torch.from_numpy(erosionmask + 1).cuda())
    sigma_gaosi1 = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0],2))*11).cuda())
    inv_soft_mask = cvcuda.gaussian(inv_mask_center,(max_ks,max_ks),kernel_size,sigma_gaosi1,cvcuda.Border.CONSTANT)

    # for bimg in inv_soft_mask:
    #     test_input_nvcv_tensor = nvcv.as_tensor(bimg)
    #     test_input_nvcv_tensor = torch.as_tensor(
    #             test_input_nvcv_tensor.cuda(), device="cuda"
    #     )
    #     img_numpy = test_input_nvcv_tensor.cpu().numpy()
    #     cv2.imwrite("./inv_soft_mask.png",img_numpy.squeeze() * 255)
    #     break
    
    ############# 慢，可以同时处理256张
    parse_mask = np.zeros(out.shape, dtype=np.float32) 
    MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
    for idx, color in enumerate(MASK_COLORMAP):
        parse_mask[out == idx] = color
    
    
    # import pdb;pdb.set_trace()
    mparse_mask_image_batch = nvcv.ImageBatchVarShape(restored_face.shape[0])
    for index in range(restored_face.shape[0]):
        input_nvcv_image = cvcuda_utils.to_nvcv_image(parse_mask[index])
        mparse_mask_image_batch.pushback(input_nvcv_image)
        
    kernel_size = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0],2), dtype=np.int32)*101).cuda())
    sigma_gaosi1 = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0],2))*11).cuda())
    mparse_mask_image_batch = cvcuda.gaussian(mparse_mask_image_batch,(101,101),kernel_size,sigma_gaosi1,cvcuda.Border.CONSTANT)
    mparse_mask_image_batch = cvcuda.gaussian(mparse_mask_image_batch,(101,101),kernel_size,sigma_gaosi1,cvcuda.Border.CONSTANT)
    # import pdb;pdb.set_trace()
    parse_mask_image_batch = nvcv.ImageBatchVarShape(restored_face.shape[0])
    for bimg in mparse_mask_image_batch:
        test_input_nvcv_tensor = nvcv.as_tensor(bimg)
        test_input_nvcv_tensor = torch.as_tensor(test_input_nvcv_tensor.cuda()).squeeze().numpy()
        test_input_nvcv_tensor[:10, :] = 0
        test_input_nvcv_tensor[-10:, :] = 0
        test_input_nvcv_tensor[:, :10] = 0
        test_input_nvcv_tensor[:, -10:] = 0
        test_input_nvcv_tensor = test_input_nvcv_tensor / 255.
        input_nvcv_image = cvcuda_utils.to_nvcv_image(test_input_nvcv_tensor)
        parse_mask_image_batch.pushback(input_nvcv_image)
    # import pdb;pdb.set_trace()
    # for bimg in parse_mask_image_batch:
    #     test_input_nvcv_tensor = nvcv.as_tensor(bimg)
    #     test_input_nvcv_tensor = torch.as_tensor(
    #             test_input_nvcv_tensor.cuda(), device="cuda"
    #     )
    #     img_numpy = test_input_nvcv_tensor.cpu().numpy()
    #     cv2.imwrite("./parse_mask_image_batch.png",img_numpy.squeeze()*255)
    #     break
    dstsize = [(512,512)] * restored_face.shape[0]
    parse_mask = cvcuda.resize(parse_mask_image_batch, dstsize)
    # import pdb;pdb.set_trace()
    # for bimg in parse_mask:
    #     test_input_nvcv_tensor = nvcv.as_tensor(bimg)
    #     test_input_nvcv_tensor = torch.as_tensor(
    #             test_input_nvcv_tensor.cuda(), device="cuda"
    #     )
    #     img_numpy = test_input_nvcv_tensor.cpu().numpy()
    #     cv2.imwrite("./parse_mask.png",img_numpy.squeeze()*255)
    #     break
    tmpsize = cvcuda_utils.clone_image_batch(parse_mask, newsize=(w,h)) 
    stream = cvcuda.Stream()
    parse_mask_image_batch = cvcuda.warp_affine_into(src=parse_mask_image_batch,dst=tmpsize,xform=cvcuda_inverse_affine_matrices,flags=cvcuda.Interp.NEAREST,
        border_mode=cvcuda.Border.CONSTANT,border_value=[0],stream=stream)
    # import pdb;pdb.set_trace()
    # for bimg in parse_mask_image_batch:
    #     test_input_nvcv_tensor = nvcv.as_tensor(bimg)
    #     test_input_nvcv_tensor = torch.as_tensor(
    #             test_input_nvcv_tensor.cuda(), device="cuda"
    #     )
    #     img_numpy = test_input_nvcv_tensor.cpu().numpy()
    #     cv2.imwrite("./parse_mask_image_batch.png",img_numpy.squeeze()*255)
    #     break
    
    output_imglist = []
    indx = 0
    import pdb;pdb.set_trace()
    for bimg, inv_soft_mask_one,past_face_one in zip(parse_mask_image_batch,inv_soft_mask,pasted_face_list):
        test_input_nvcv_tensor = nvcv.as_tensor(bimg)
        test_input_nvcv_tensor = torch.as_tensor(
                test_input_nvcv_tensor.cuda(), device="cuda"
        )
        cv2.imwrite('parse_mask_image_batch.png', test_input_nvcv_tensor.squeeze().cpu().numpy()*255)
        inv_soft_mask_one = torch.as_tensor(nvcv.as_tensor(inv_soft_mask_one).cuda(), device="cuda").squeeze()[:, :, None]
        cv2.imwrite('inv_soft_mask.png', inv_soft_mask_one.squeeze().cpu().numpy()*255)
        inv_soft_parse_mask = test_input_nvcv_tensor.squeeze()[:, :, None]
        fuse_mask = (inv_soft_parse_mask<inv_soft_mask_one)
        fuse_mask = torch.tensor(fuse_mask,dtype=torch.int)
        cv2.imwrite('fuse_mask.png', fuse_mask.squeeze().cpu().numpy()*255)
        inv_soft_mask_one = inv_soft_parse_mask*fuse_mask + inv_soft_mask_one*(1-fuse_mask)
        upsample_img = inv_soft_mask_one * past_face_one.cuda() + (1 - inv_soft_mask_one) * torch.tensor(imglist[indx]).cuda()
        output_img = upsample_img.cpu().numpy().astype(np.uint16)
        indx += 1
        output_imglist.append()
    # inv_soft_parse_mask = parse_mask[:, :, None]
    # fuse_mask = (inv_soft_parse_mask<inv_soft_mask).astype('int')
    # inv_soft_mask = ne.evaluate('inv_soft_parse_mask*fuse_mask + inv_soft_mask*(1-fuse_mask)')
    # upsample_img = ne.evaluate('inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img')
    return output_imglist
    ################################################################################



# 改背景，加logo之类的工作，第一版不用做
def run_postprocessing(imglist, fps, audio_path, save_video_path):
    '''
        后处理：改背景、加logo、拼音频等
    :param output_imglist: 图像优化后的imglist
    :param audio_path: 音频文件
    :param final_output_path: 最终的视频文件
    :return: final_output_path 不带音频的视频暂存路径
    '''
    imageio.mimsave(save_video_path, [img[:, :, ::-1] for img in imglist], fps=fps)    # 保存成视频

    # # 也可以通过这种方式保存
    # with imageio.get_writer(final_output_path, fps=fps) as video:
    #     for img in imglist:
    #         video.append_data(img)

    audio_clip = AudioFileClip(audio_path)
    video_clip = VideoFileClip(save_video_path)

    final_clip = video_clip.set_audio(audio_clip)    # 合并
    final_output_path = save_video_path[0:-4] + "_final.mp4"
    final_clip.write_videofile(final_output_path)

    return final_output_path


if __name__ == "__main__":
    
    
    """第二种方式：读取文件夹中的图像文件，生成一个imagelist
    """
    # 图片文件夹的路径
    folder_path = "/path/to/your/folder"

    imagelist = []  # 创建一个空的imagelist

    # 遍历文件夹中的图片文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 使用OpenCV读取图像
        image = cv2.imread(file_path)
        
        if image is not None:
            # 调整图像大小为512 * 512
            image = cv2.resize(image, (512, 512))
            
            # 将图像转换为numpy数组，并添加到imagelist中
            image_array = np.array(image)
            imagelist.append(image_array)

    # 将imagelist转换为numpy数组
    imagelist = np.array(imagelist)

    # 输出imagelist的形状
    print(imagelist.shape)
    
    # 图像中拆出人脸--->人脸修复/超分--->人脸拼接回原图操作
    output_imagelist = run_inpainting(imagelist)
    
    # 后处理：将output_imagelist转换为视频
    run_postprocessing(output_imagelist, fps=25, audio_path="/path/to/your/audio", save_video_path="/path/to/your/video")
