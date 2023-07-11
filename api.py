import os
import subprocess
import cv2
import numpy as np
import imageio
import traceback
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip

import config as cfg
from config import log, params
from Inference_wav2lip_tools import get_video_frame_list, get_audio_frame_list, datagen, load_model
from Inference_inpainting_tools import set_realesrgan

from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

class TalkingLipPipeline:
    def __init__(self):
        self.wav2lip = None
        self.inpainting = None    # 使用codeformer做图像修复


    def run_wav2lip(self, audio_path, video_source_path):
        video_frame_list, fps = get_video_frame_list(video_source_path)
        mel_chunks = get_audio_frame_list(audio_path, fps)

        video_frame_list = video_frame_list[: len(mel_chunks)]  # 只截取和audio相同长度的video

        batch_size = params['app']['wav2lip']['wav2lip_batch_size']
        gen = datagen(video_frame_list.copy(), mel_chunks)

        model = load_model(params['app']['wav2lip']['checkpoint_path'])
        log.logger.info("wav2lip Model loaded")

        wav2lip_result = []
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(cfg.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(cfg.device)

            # log.logger.info("img_batch.shape: %s" % str(img_batch.shape))
            # log.logger.info("mel_batch.shape: %s" % str(mel_batch.shape))

            with torch.no_grad():
                pred = model(mel_batch, img_batch)  # [batch_size, 3, 96, 96] bchw

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.  # 这里直接放在gpu上

            # 循环拼图
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c

                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                wav2lip_result.append(f)

                # cv2.imshow("f", f)    # 全身照
                # cv2.waitKey()
                #
                # cv2.imshow("p", p)    # 人脸照
                # cv2.waitKey()

            # wav2lip_result.extend(frames)
        return wav2lip_result, fps

    def run_inpainting(self, imglist):
        # bg_upsampler = set_realesrgan()
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
        for frame_idx in tqdm(range(len(imglist))):
            img = imglist[frame_idx]
            face_helper.clean_all()
            face_helper.read_image(img)

            # start_time = time.time()
            num_det_faces = face_helper.get_face_landmarks_5(only_center_face=params['app']['inpainting']['only_center_face'], resize=640, eye_dist_threshold=5)
            # log.logger.info("[%d / %d] detect %s faces" % (frame_idx, len(imglist), str(num_det_faces)))
            # print("获取关键点耗时：", time.time() - start_time)
            # start_time = time.time()

            # align and warp each face
            face_helper.align_warp_face()  # 修复前对齐
            # print("修复前对齐：", time.time() - start_time)
            # start_time = time.time()

            #
            # face restoration for each cropped face
            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                # prepare data
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(cfg.device)

                try:
                    with torch.no_grad():
                        output = net(cropped_face_t, w=params['app']['inpainting']['fidelity_weight'], adain=True)[0]    # [1, 3, 512, 512]
                        output_copy = output.clone()

                        restored_face = tensor2img(output_copy, rgb2bgr=True, min_max=(-1, 1))  # restored_face [512, 512, 3]

                        # 分割
                        # inference 分割网络放到外面
                        # face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                        # face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                        normalize(output, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                        # face_input = torch.unsqueeze(face_input, 0).to(cfg.device)
                        with torch.no_grad():
                            out = face_helper.face_parse(output)[0]

                        out = out.argmax(dim=1).squeeze().cpu().numpy()    # [512, 512]

                        # print("restored_face.shape:", restored_face.shape)
                        #
                        # cv2.imshow("face_parse_result", out)
                        # cv2.waitKey()
                        # print("脸部修复推理之后：", time.time() - start_time)
                        # start_time = time.time()


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
                    # start_time = time.time()
                    bg_img = bg_upsampler.enhance(img, outscale=params['app']['inpainting']['upscale'])[0]
                    # print("背景超分耗时：", time.time() - start_time)
                    # start_time = time.time()
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)
                # paste each restored face to the input image

                if params['app']['inpainting']['face_upsample'] and face_upsampler is not None:
                    restored_img = face_helper.paste_faces_to_input_image(face_parse_out=out, upsample_img=bg_img, draw_box=params['app']['inpainting']['draw_box'],
                                                                          face_upsampler=face_upsampler)
                else:
                    restored_img = face_helper.paste_faces_to_input_image(face_parse_out=out, upsample_img=bg_img, draw_box=params['app']['inpainting']['draw_box'])
                output_imglist.append(restored_img)
                
                            
                # print("all_landmarks_5:",type(face_helper.all_landmarks_5))
                # print("face_helper.det_faces:",type(face_helper.det_faces))
                # print("affine_matrices:",type(face_helper.affine_matrices))
                # print("inverse_affine_matrices:",type(face_helper.inverse_affine_matrices))
                
                # print("贴回去：", time.time() - start_time)
        return output_imglist



    # 改背景，加logo之类的工作，第一版不用做
    def run_postprocessing(self, imglist, fps, audio_path, save_video_path):
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
        # final_clip.write_videofile(final_output_path)
        ## 若以上语句拼音频无效，直接使用以下句子调用ffmpeg拼音频视频 added by dengjunli
        subprocess.call(f"ffmpeg -i %s -i %s -y %s" % (audio_path, save_video_path, final_output_path),shell=True)


        return final_output_path


