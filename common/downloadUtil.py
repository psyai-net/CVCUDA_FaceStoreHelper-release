import os
import requests
import traceback

# import config as cfg
from config import log, params


headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36',
    'referer': 'https://space.bilibili.com/28152637/channel/seriesdetail?sid=1259590'    # 跳过防盗链
}


'''
    请求数据
'''
def send_request(url):
    response = requests.get(url=url, headers=headers)
    return response


'''
    从指定url下载文件
'''
def get_file_from_url(audio_url, task_id):
    save_path = os.path.join("%s/%s.wav" % (params['app']['audio_input_path'], task_id))

    log.logger.info('正在请求音频数据: %s' % audio_url)
    try:
        # audio_data = send_request(audio_url).content
        #
        # with open(save_path, mode='wb') as f:
        #     f.write(audio_data)

        COMMAND = "wget %s -O %s" % (audio_url, save_path)
        os.system(COMMAND)

        log.logger.info("COMMAND: %s" % COMMAND)
        log.logger.info("音频数据保存完成：%s" % save_path)
        return save_path
    except Exception as e:
        log.logger.error(traceback.format_exc())
        log.logger.info("请求音频数据失败：%s" % traceback.format_exc())
        return None





