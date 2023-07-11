import os
import time
import json
import datetime
import traceback

import api
import config as cfg
from config import log, params, task_info_dict
from common.dbUtil import save_progress, update_task_info
from common.downloadUtil import get_file_from_url
from common.uploadUtil import put_video_to_cloud
from common.progressUtil import Progress, Progress_entity




'''
    生成video的方法：子进程调用
    :param task_id 任务ID
    :param data 其他参数
'''
def gen_video(task_id, data):
    cfg.running = True    # 表示目前有任务在执行

    talking_lip_pipeline = api.TalkingLipPipeline()  # 流程pipeline
    progress = Progress()  # 进度工具类

    save_video_path = params['app']['final_output_path'] + "%s.mp4" % task_id

    video_source_path = params['app']['video_based_path'] + str(data['Payload']['VirtualmanKey']) + ".mp4"    # video_based path

    # 第一版这些不必要
    # background_image = get_file_from_url(data['Payload']['VideoParam']['BackgroundFileUrl'])    # 背景图下载
    # logo_image = get_file_from_url(data['Payload']['VideoParam']['LogoParams'][0]['LogoFileUrl'])    # logo文件下载
    #
    # video_params = {
    #     'background': background_image,
    #     'logo_params': {
    #         'logo': logo_image,
    #         'x': data['Payload']['VideoParam']['LogoParams'][0]['PositionX'],
    #         'y': data['Payload']['VideoParam']['LogoParams'][0]['PositionY'],
    #         'scale': data['Payload']['VideoParam']['LogoParams'][0]['Scale']
    #     },
    #     'anchor_params': {
    #         'x': data['Payload']['VideoParam']['AnchorParam']['HorizontalPosition'],
    #         'scale': data['Payload']['VideoParam']['AnchorParam']['Scale']
    #     }
    # }

    # step0: 提交任务
    task_info_dict[task_id] = Progress_entity(status="COMMIT", info="任务已提交，请稍后", progress=0, MediaURL="")

    # step1：下载audio文件
    stage_name = "download_audio"
    # audio_path = get_file_from_url(audio_url=data['Payload']['Audio_url'], task_id=task_id)  # 微软tts生成，我们下载（文件下载）
    #
    # if audio_path is None:
    #     task_info_dict[task_id] = Progress_entity(status="FAIL", info="音频文件下载失败", progress=0, MediaURL="")

    audio_path = "./testdata/audio_input/123.mp3"

    # step1: audio文件已下载
    task_info_dict[task_id] = Progress_entity(status="MAKING", info='audio文件已下载，保存为：%s' % audio_path, progress=progress.get_progress(stage_name=stage_name), MediaURL="")

    # step2：wav2lip
    stage_name = "wav2lip"
    log.logger.info("wav2lip start...")
    wav2lip_result = []
    fps = 60    # 默认为60
    try:
        wav2lip_result, fps = talking_lip_pipeline.run_wav2lip(audio_path, video_source_path)
        task_info_dict[task_id] = Progress_entity(status="MAKING",
                                                  info='wav2lip已完成',
                                                  progress=progress.get_progress(stage_name=stage_name),
                                                  MediaURL="")

        log.logger.info("wav2lip finished")
    except Exception as e:
        log.logger.error(traceback.format_exc())
        log.logger.error("run_wav2lip exception occurred: %s" % traceback.format_exc())

        task_info_dict[task_id] = Progress_entity(status="FAIL",
                                                  info='wav2lip端出现异常，详情请查看日志',
                                                  progress=progress.get_progress(stage_name=stage_name),
                                                  MediaURL="")
        cfg.running = False

    # step4：图像修复：inpainting
    stage_name = "inpainting"
    log.logger.info("inpainting start...")
    output_imglist = []    # 图像修复后的列表
    try:
        output_imglist = talking_lip_pipeline.run_inpainting(wav2lip_result)
        task_info_dict[task_id] = Progress_entity(status="MAKING",
                                                  info='inpainting已完成',
                                                  progress=progress.get_progress(stage_name=stage_name),
                                                  MediaURL="")
        log.logger.info("inpainting finished")
    except Exception as e:
        log.logger.error(traceback.format_exc())
        log.logger.error("inpainting exception occurred: %s" % traceback.format_exc())

        task_info_dict[task_id] = Progress_entity(status="FAIL",
                                                  info='inpainting端出现异常，详情请查看日志',
                                                  progress=progress.get_progress(stage_name=stage_name),
                                                  MediaURL="")
        cfg.running = False

    # step5：后处理：post_processing端，添加背景图、logo等、拼接音频等操作
    stage_name = "post_processing"
    log.logger.info("post processing start...")
    try:
        save_video_final_path = talking_lip_pipeline.run_postprocessing(output_imglist, fps, audio_path, save_video_path)
        task_info_dict[task_id] = Progress_entity(status="MAKING",
                                                  info='post_processing已完成',
                                                  progress=progress.get_progress(stage_name=stage_name),
                                                  MediaURL="")

        log.logger.info("post processing finished")
    except Exception as e:
        log.logger.error(traceback.format_exc())
        log.logger.error("post processing exception occurred: %s" % traceback.format_exc())

        task_info_dict[task_id] = Progress_entity(status="FAIL",
                                                  info='post_processing端出现异常，详情请查看日志',
                                                  progress=progress.get_progress(stage_name=stage_name),
                                                  MediaURL="")
        cfg.running = False

    # step6：上传生成的视频
    stage_name = "put_to_cloud"
    log.logger.info("upload file to server: %s" % save_video_final_path)
    try:
        video_url = put_video_to_cloud(save_video_final_path)
        task_info_dict[task_id] = Progress_entity(status="SUCCESS",
                                                  info='上传视频文件到云端成功',
                                                  progress=progress.get_progress(stage_name=stage_name),
                                                  MediaURL=video_url)

        log.logger.info("upload file to server finished: %s" % save_video_final_path)
    except Exception as e:
        log.logger.error(traceback.format_exc())
        log.logger.error("put_video_to_cloud exception occurred: %s" % traceback.format_exc())

        task_info_dict[task_id] = Progress_entity(status="FAIL",
                                                  info='上传视频文件到服务器出现异常，详情请查看日志',
                                                  progress=progress.get_progress(stage_name=stage_name),
                                                  MediaURL="")
        cfg.running = False

    # 视频生成完成之后，将标识设为False
    cfg.running = False


















