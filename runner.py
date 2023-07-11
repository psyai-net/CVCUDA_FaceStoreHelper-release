import os
import time
import numpy as np

import api

'''
    本地调试用
'''

start_time = time.time()

talking_lip_pipeline = api.TalkingLipPipeline()  # 流程pipeline



audio_path = "./帮范肇心跑demo/5_优尼康集团简介_(Vocals).wav"
# video_source_path = "./testdata/video_based/zixin/zixin.mp4"
video_source_path = "./帮范肇心跑demo/说话.mp4"

wav2lip_result, fps = talking_lip_pipeline.run_wav2lip(audio_path, video_source_path)

output_imglist = talking_lip_pipeline.run_inpainting(wav2lip_result)

# 保存最终输出视频
save_video_path = "./帮范肇心跑demo/说话-尼康集团简介.mp4"

save_video_final_path = talking_lip_pipeline.run_postprocessing(output_imglist, fps, audio_path, save_video_path)

print(save_video_final_path)
print("全流程耗时：", time.time() - start_time)




audio_path = "./帮范肇心跑demo/5_优尼康集团简介_(Vocals).wav"
# video_source_path = "./testdata/video_based/zixin/zixin.mp4"
video_source_path = "./帮范肇心跑demo/不说话.mp4"

wav2lip_result, fps = talking_lip_pipeline.run_wav2lip(audio_path, video_source_path)

output_imglist = talking_lip_pipeline.run_inpainting(wav2lip_result)

# 保存最终输出视频
save_video_path = "./帮范肇心跑demo/不说话-尼康集团简介.mp4"

save_video_final_path = talking_lip_pipeline.run_postprocessing(output_imglist, fps, audio_path, save_video_path)

print(save_video_final_path)
print("全流程耗时：", time.time() - start_time)






audio_path = "./帮范肇心跑demo/2_主营业务_(Vocals).wav"
# video_source_path = "./testdata/video_based/zixin/zixin.mp4"
video_source_path = "./帮范肇心跑demo/说话.mp4"

wav2lip_result, fps = talking_lip_pipeline.run_wav2lip(audio_path, video_source_path)

output_imglist = talking_lip_pipeline.run_inpainting(wav2lip_result)

# 保存最终输出视频
save_video_path = "./帮范肇心跑demo/说话-主营业务.mp4"

save_video_final_path = talking_lip_pipeline.run_postprocessing(output_imglist, fps, audio_path, save_video_path)

print(save_video_final_path)
print("全流程耗时：", time.time() - start_time)



audio_path = "./帮范肇心跑demo/2_主营业务_(Vocals).wav"
# video_source_path = "./testdata/video_based/zixin/zixin.mp4"
video_source_path = "./帮范肇心跑demo/不说话.mp4"

wav2lip_result, fps = talking_lip_pipeline.run_wav2lip(audio_path, video_source_path)

output_imglist = talking_lip_pipeline.run_inpainting(wav2lip_result)

# 保存最终输出视频
save_video_path = "./帮范肇心跑demo/不说话-主营业务.mp4"

save_video_final_path = talking_lip_pipeline.run_postprocessing(output_imglist, fps, audio_path, save_video_path)

print(save_video_final_path)
print("全流程耗时：", time.time() - start_time)



