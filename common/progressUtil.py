import os

'''
    进度工具类
'''

class Progress():
    def __init__(self):
        self.stage_list = ['download_audio', 'wav2lip', 'inpainting', 'post_processing', 'put_to_cloud']

    def get_progress(self, stage_name):
        return ((self.stage_list.index(stage_name) + 1) / len(self.stage_list)) * 100.0

'''
    进度的实体类
'''
class Progress_entity():
    def __init__(self, status="COMMIT", info="", progress=0, MediaURL=""):
        self.status = status      # 当前状态：COMMIT已提交；MAKING制作中；SUCCESS制作成功；FAIL制作失败
        self.info = info          # 进度信息
        self.progress = progress  # 进度值
        self.MediaURL = MediaURL  # 生成视频的URL





if __name__ == '__main__':
    progress = Progress()
    print(progress.get_progress(stage_name="wav2lip"))
