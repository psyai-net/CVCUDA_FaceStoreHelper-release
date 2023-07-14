import os
import cv2
import numpy as np
import imageio
import traceback
import torch
import pickle
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from basicsr.utils.video_util import VideoReader, VideoWriter

import config as cfg
from config import log, params

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


def collate_fn(batch):
    img_batch = [item[0] for item in batch]
    mel_batch = [item[1] for item in batch]
    frame_batch = [item[2] for item in batch]
    coords_batch = [item[3] for item in batch]

    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

    img_masked = img_batch.copy()
    img_masked[:, params['app']['wav2lip']['img_size'] // 2:] = 0

    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    return img_batch, mel_batch, frame_batch, coords_batch


def load_all_models(resource_path):
    import onnxruntime
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else [
        'CPUExecutionProvider']
    model_face_parser = onnxruntime.InferenceSession(params['app']['inpainting']['face_parse_checkpoint_path'],
                                                          providers=providers)

    wav2lip_model = load_model(params['app']['wav2lip']['checkpoint_path'])
    log.logger.info("wav2lip Model loaded")
    codeformer_net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                                          connect_list=['32', '64', '128', '256']).to(cfg.device)

    checkpoint = torch.load(params['app']['inpainting']['codeformer_checkpoint_path'])['params_ema']
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()
    log.logger.info("Codeformer Model loaded")

    with open(str(resource_path), "rb") as f:
        resource = pickle.load(f)
    log.logger.info("使用本地资源包 skipping face_detection")

    with open(params['app']['codeformer_source_path'] + 'affine_matrices.pkl', 'rb') as f3:
        affine_matrices = pickle.load(f3)
    with open(params['app']['codeformer_source_path'] + 'inverse_affine_matrices.pkl', 'rb') as f4:
        inverse_affine_matrices = pickle.load(f4)
    log.logger.info("通过本地资源包读取人脸正脸化需要的数据")

    return model_face_parser, wav2lip_model, codeformer_net, resource, affine_matrices, inverse_affine_matrices



def run_inpainting_1(imglist, affine_matrices, inverse_affine_matrices):
    # ------------------ set up CodeFormer restorer -------------------
    output_imglist = []

    affine_matrices = [item[0].tolist() for item in affine_matrices]
    affine_matrices = np.array(affine_matrices, dtype=np.float32)[:len(imglist)].reshape(-1, 6)
    inverse_affine_matrices = [item[0].tolist() for item in inverse_affine_matrices]
    inverse_affine_matrices = np.array(inverse_affine_matrices, dtype=np.float32)[:len(imglist)].reshape(-1, 6)
    cvcuda_affine_matrices = cvcuda_utils.to_nvcv_tensor(affine_matrices, "NC")
    cvcuda_inverse_affine_matrices = cvcuda_utils.to_nvcv_tensor(inverse_affine_matrices, "NC")

    image_batch = nvcv.ImageBatchVarShape(len(imglist))
    # 0.44s
    for index in range(len(imglist)):
        input_nvcv_image = cvcuda_utils.to_nvcv_image(imglist[index])
        image_batch.pushback(input_nvcv_image)
    out = cvcuda.warp_affine(
        image_batch, cvcuda_affine_matrices, cvcuda.Interp.LINEAR, border_mode=cvcuda.Border.CONSTANT,
        border_value=[0]
    )
    cropped_face_t_list = []
    for bimg in out:
        test_input_nvcv_tensor = nvcv.as_tensor(bimg)
        test_input_nvcv_tensor = torch.as_tensor(test_input_nvcv_tensor.cuda(), device="cuda")
        immm = test_input_nvcv_tensor[:, :512, :512, :].squeeze().cpu().numpy()
        cropped_face_t = img2tensor(immm / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t_list.append(cropped_face_t)

    cropped_face_all = torch.stack(cropped_face_t_list).to(cfg.device)
    return cropped_face_all, cvcuda_inverse_affine_matrices


def run_inpainting_2(cropped_face_all):
    
    # ------------------ set up CodeFormer restorer -------------------
    codeformer_net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                            connect_list=['32', '64', '128', '256']).to(cfg.device)

    checkpoint = torch.load(params['app']['inpainting']['codeformer_checkpoint_path'])['params_ema']
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()

    face_helper = FaceRestoreHelper(
        params['app']['inpainting']['upscale'],
        face_size=512,
        crop_ratio=(1, 1),
        det_model=params['app']['inpainting']['detection_model'],
        save_ext='png',
        use_parse=True,
        device=cfg.device)  # 人脸关键点检测用的是 retinaface_resnet50
    
    ################### 第二个block
    all_images = cropped_face_all.shape[0]
    print('all_images number: ', all_images)
    netbatchsize = 10
    # import pdb;pdb.set_trace()
    restored_face = []
    out = []
    for i in range(0, all_images, netbatchsize):
        if (i + 8) < all_images:
            cropped_face_t = cropped_face_all[i:i + 8]
        else:
            cropped_face_t = cropped_face_all[i:]
        with torch.no_grad():
            output = codeformer_net(cropped_face_t, w=params['app']['inpainting']['fidelity_weight'], adain=True)[
                0]  # [1, 3, 512, 512]
            output_copy = output.clone()  # torch.Size([12, 3, 512, 512])
            # tensor 2 img
            min_max = (-1, 1)
            output_copy = output_copy.float().detach().cpu().clamp_(*min_max)
            output_copy = (output_copy - min_max[0]) / (min_max[1] - min_max[0])
            output_copy = output_copy.numpy().transpose(0, 2, 3, 1)
            # output_copy = cv2.cvtColor(output_copy, cv2.COLOR_RGB2BGR)
            output_copy_orl = output_copy.copy()
            output_copy[:, :, :, 0] = output_copy_orl[:, :, :, 2]
            output_copy[:, :, :, 2] = output_copy_orl[:, :, :, 0]
            restored_face_tmp = (output_copy * 255.0).round().astype(np.uint8)
            normalize(output, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            with torch.no_grad():
                ''' onnx 版本 '''
                # out_tmp = model_face_parser.run(
                #     None,
                #     {"onnx::Pad_0": output.cpu().numpy()},
                # )[0]
                # out_tmp = np.argmax(out_tmp, axis=1).squeeze()
                
                '''pth版本'''
                out_tmp = face_helper.face_parse(output)[0] # torch.Size([12, 19, 512, 512])
                out_tmp = out_tmp.argmax(dim=1).squeeze().cpu().numpy() # (12, 512, 512)
                
            restored_face.append(restored_face_tmp)
            out.append(out_tmp)
        del output
        torch.cuda.empty_cache()
    restored_face = np.concatenate(restored_face, axis=0)
    out = np.concatenate(out, axis=0)

    return out, restored_face


def run_inpainting_3(out, imglist, restored_face, cvcuda_inverse_affine_matrices):
    ################### 第三个block
    ### 经过codeformer和parse两个网络结束，转成cvcuda格式继续后面的cv2加速
    h, w, _ = imglist[0].shape

    restore_image_batch = nvcv.ImageBatchVarShape(restored_face.shape[0])
    restored_face = np.ascontiguousarray(restored_face)
    for index in range(restored_face.shape[0]):
        input_nvcv_image = cvcuda_utils.to_nvcv_image(restored_face[index])
        restore_image_batch.pushback(input_nvcv_image)
    tmpsize = cvcuda_utils.clone_image_batch(restore_image_batch, newsize=(w, h))
    stream = cvcuda.Stream()
    inv_restored = cvcuda.warp_affine_into(src=restore_image_batch, dst=tmpsize,
                                           xform=cvcuda_inverse_affine_matrices, flags=cvcuda.Interp.LINEAR,
                                           border_mode=cvcuda.Border.CONSTANT, border_value=[0], stream=stream)

    mask = np.ones((restored_face.shape[0], out.shape[-1], out.shape[-1]), dtype=np.float32)
    mask_image_batch = nvcv.ImageBatchVarShape(restored_face.shape[0])
    for index in range(restored_face.shape[0]):
        input_nvcv_image = cvcuda_utils.to_nvcv_image(mask[index])
        mask_image_batch.pushback(input_nvcv_image)
    tmpsize = cvcuda_utils.clone_image_batch(mask_image_batch, newsize=(w, h))

    stream = cvcuda.Stream()
    inv_mask = cvcuda.warp_affine_into(src=mask_image_batch, dst=tmpsize, xform=cvcuda_inverse_affine_matrices,
                                       flags=cvcuda.Interp.LINEAR,
                                       border_mode=cvcuda.Border.CONSTANT, border_value=[0], stream=stream)
    # for bimg in inv_mask:
    #     test_input_nvcv_tensor = nvcv.as_tensor(bimg)
    erosionmask = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0], 2), np.int32) * 2).cuda())
    anchor = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0], 2), np.int32) * -1).cuda())
    inv_mask_erosion = cvcuda.morphology(inv_mask, cvcuda.MorphologyType.ERODE, erosionmask, anchor,
                                         border=cvcuda.Border.CONSTANT)

    erosionmask = np.ones((restored_face.shape[0], 2), np.int32)
    idx = 0
    pasted_face_list = []
    for inv_mask, inv_restore_img in zip(inv_mask_erosion, inv_restored):
        tmpp_inv_mask = torch.as_tensor(nvcv.as_tensor(inv_mask).cuda()).squeeze()[:, :, None]
        tmpp_inv_restore_img = torch.as_tensor(nvcv.as_tensor(inv_restore_img).cuda()).squeeze()
        pasted_face = tmpp_inv_mask * tmpp_inv_restore_img
        pasted_face_list.append(pasted_face)
        total_face_area = np.sum(tmpp_inv_mask.cpu().numpy())
        erosion_radius = (int(total_face_area ** 0.5) // 20) * 2
        erosionmask[idx] *= erosion_radius
        idx += 1

    erosionmask_1 = nvcv.as_tensor(torch.from_numpy(erosionmask).cuda())
    inv_mask_center = cvcuda.morphology(inv_mask_erosion, cvcuda.MorphologyType.ERODE, erosionmask_1, anchor,
                                        border=cvcuda.Border.CONSTANT)

    max_ks = np.max(erosionmask) + 1
    kernel_size = nvcv.as_tensor(torch.from_numpy(erosionmask + 1).cuda())
    sigma_gaosi1 = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0], 2)) * 11).cuda())
    inv_soft_mask = cvcuda.gaussian(inv_mask_center, (max_ks, max_ks), kernel_size, sigma_gaosi1,
                                    cvcuda.Border.CONSTANT)

    #######
    parse_mask = np.zeros(out.shape, dtype=np.float32)
    MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
    for idx, color in enumerate(MASK_COLORMAP):
        parse_mask[out == idx] = color

    mparse_mask_image_batch = nvcv.ImageBatchVarShape(restored_face.shape[0])
    print("parse_mask.shape:", parse_mask.shape)    # parse_mask.shape: (40, 512, 512)
    print("parse_mask[0].shape:", parse_mask[0].shape)    # parse_mask[0].shape: (512, 512)
    print("restored_face.shape:", restored_face.shape)    # restored_face.shape: (40, 512, 512, 3)
    for index in range(restored_face.shape[0]):
        input_nvcv_image = cvcuda_utils.to_nvcv_image(np.ascontiguousarray(parse_mask[index]))
        mparse_mask_image_batch.pushback(input_nvcv_image)

    kernel_size = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0], 2), dtype=np.int32) * 101).cuda())
    sigma_gaosi1 = nvcv.as_tensor(torch.from_numpy(np.ones((restored_face.shape[0], 2)) * 11).cuda())
    mparse_mask_image_batch = cvcuda.gaussian(mparse_mask_image_batch, (101, 101), kernel_size, sigma_gaosi1,
                                              cvcuda.Border.CONSTANT)
    mparse_mask_image_batch = cvcuda.gaussian(mparse_mask_image_batch, (101, 101), kernel_size, sigma_gaosi1,
                                              cvcuda.Border.CONSTANT)

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

    dstsize = [(512, 512)] * restored_face.shape[0]
    parse_mask = cvcuda.resize(parse_mask_image_batch, dstsize)

    tmpsize = cvcuda_utils.clone_image_batch(parse_mask, newsize=(w, h))
    stream = cvcuda.Stream()
    parse_mask_image_batch = cvcuda.warp_affine_into(src=parse_mask_image_batch, dst=tmpsize,
                                                     xform=cvcuda_inverse_affine_matrices,
                                                     flags=cvcuda.Interp.LINEAR,
                                                     border_mode=cvcuda.Border.CONSTANT, border_value=[0],
                                                     stream=stream)

    output_imglist = []
    indx = 0
    # import pdb;pdb.set_trace()
    for bimg, inv_soft_mask_one, past_face_one in zip(parse_mask_image_batch, inv_soft_mask, pasted_face_list):
        test_input_nvcv_tensor = nvcv.as_tensor(bimg)
        test_input_nvcv_tensor = torch.as_tensor(
            test_input_nvcv_tensor.cuda(), device="cuda"
        )
        # cv2.imwrite('parse_mask_image_batch.png', test_input_nvcv_tensor.squeeze().cpu().numpy()*255)
        inv_soft_mask_one = torch.as_tensor(nvcv.as_tensor(inv_soft_mask_one).cuda(), device="cuda").squeeze()[:, :,
                            None]
        # cv2.imwrite('inv_soft_mask.png', inv_soft_mask_one.squeeze().cpu().numpy()*255)
        inv_soft_parse_mask = test_input_nvcv_tensor.squeeze()[:, :, None]
        fuse_mask = (inv_soft_parse_mask < inv_soft_mask_one)
        fuse_mask = torch.tensor(fuse_mask, dtype=torch.int)
        # cv2.imwrite('fuse_mask.png', fuse_mask.squeeze().cpu().numpy()*255)
        inv_soft_mask_one = inv_soft_parse_mask * fuse_mask + inv_soft_mask_one * (1 - fuse_mask)
        upsample_img = inv_soft_mask_one * past_face_one.cuda() + (1 - inv_soft_mask_one) * torch.tensor(
            imglist[indx]).cuda()
        output_img = upsample_img.cpu().numpy().astype(np.uint8)
        indx += 1
        output_imglist.append(output_img)

    return output_imglist
    ################################################################################


# 改背景，加logo之类的工作，第一版不用做
def run_postprocessing(imglist, fps, save_video_path, origin_video_path):
    # '''
    #     后处理：改背景、加logo、拼音频等
    # :param output_imglist: 图像优化后的imglist
    # :param audio_path: 音频文件
    # :param final_output_path: 最终的视频文件
    # :return: final_output_path 不带音频的视频暂存路径
    # '''
    imageio.mimsave(save_video_path, [img[:, :, ::-1] for img in imglist], fps=fps)  # 保存成视频
    
    
    final_clip = VideoFileClip(save_video_path)
    final_clip.set_audio(VideoFileClip(origin_video_path).audio) # 将原视频的音频拼回去 
    final_output_path = save_video_path[0:-4] + "_final.mp4"
    final_clip.write_videofile(final_output_path)
    
    return final_output_path

        


