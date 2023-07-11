import os
import argparse
import json
import time
import traceback
from flask import Flask, request
from threading import Thread

import config as cfg
from config import log, task_info_dict
from common.progressUtil import Progress_entity

from main import gen_video


app = Flask(__name__)


'''
    生成talking head视频
    post报文样例：
    
    {
       "Header": {},
       "Payload": {
           "VirtualmanKey": "123",
           "Audio_url": "这里是音频的url",
           "VideoParam": {
               "BackgroundFileUrl": "背景图的URL",
               "LogoParams": [
                    {
                        "LogoFileUrl":"logo图片的URL",
                        "PositionX": 100,
                        "PositionY": 100,
                        "Scale": 1.0
                    }
               ],
               "AnchorParam": {
                    "HorizontalPosition":0.5,
                    "Scale": 1.0
               }
           }
        }
    }
    
    
'''
@app.route("/gen_talking_head_video", methods=['GET', 'POST'])
def gen_talking_head_video():
    data = request.get_data()
    data = data.decode('utf-8')
    data = json.loads(data)

    log.logger.info("api:gen_talking_head_video 接收到的json报文：%s" % data)

    # 如果系统正忙，直接返回task_id为""（空字符串）
    if cfg.running is True:
        code = 101
        Message = "正在处理其他任务，请稍后。。。"
        task_id = ""

        ret_msg = '''
                    {
                       "Header": {
                           "Code":%d,
                           "DialogID":"",
                           "Message":"%s",
                           "RequestID":"123",
                        },
                        "Payload":{
                            "TaskID": "%s"
                        }
                    }
                ''' % (code, Message, task_id)
        log.logger.info("返回报文：%s" % ret_msg)
        return ret_msg
    else:    # 系统不忙的话，启动新线程生成视频
        code = 0
        Message = ""
        task_id = "%s_%s" % (data['Payload']['VirtualmanKey'], time.time())

        ret_msg = '''
                    {
                       "Header": {
                           "Code":%d,
                           "DialogID":"%s",
                           "Message":"",
                           "RequestID":"123",
                        },
                        "Payload":{
                            "TaskID": "%s"
                        }
                    }
                ''' % (code, Message, task_id)

        if len(task_id) > 0:
            task_info_dict[task_id] = Progress_entity(status="COMMIT", info="任务已提交，请稍后", progress=0, MediaURL="")

        # 这里创建个子进程，去生成视频
        task_thread = Thread(target=gen_video, args=(task_id, data, ))
        task_thread.start()

        log.logger.info("返回报文：%s" % ret_msg)
        return ret_msg



'''
    查询生成talking head视频的进度
    post报文样例：
    
    {
       "Header": {},
       "Payload": {
           "TaskId": "123"
        }
    }
'''
@app.route("/get_gen_talking_head_progress", methods=['GET', 'POST'])
def get_gen_talking_head_progress():
    data = request.get_data()
    data = data.decode('utf-8')
    data = json.loads(data)

    log.logger.info("api:get_gen_talking_head_progress 接收到的json报文：%s" % data)

    task_id = data['Payload']['TaskId']

    try:
        status = task_info_dict[task_id].status
        progress = task_info_dict[task_id].progress
        video_url = task_info_dict[task_id].MediaURL
    except KeyError as e:
        log.logger.error(traceback.format_exc())
        log.logger.error("无当前task_id信息：%s, 详情：%s" % (task_id, traceback.format_exc()))

        status = "FAIL"
        progress = 0
        video_url = ""

    ret_msg = '''
        {
           "Header": {
               "Code": 0,
               "Message": "",
               "RequestID": "123",
           },
           "Payload": {
               "Progress": %d,
               "MediaUrl": "%s",
               "SubtitlesUrl":"",
               "Status": "%s",
               "ArrayCount": 0
            }
        }
    ''' % (progress, video_url, status)

    log.logger.info("返回报文：%s" % ret_msg)
    return ret_msg

if __name__ == '__main__':

    # 这里允许指定端口
    parser = argparse.ArgumentParser(description='talking head service')
    parser.add_argument('--host', type=str, help='the local ip', default='localhost')
    parser.add_argument('--port', type=int, help='the binding port', default=5000)
    args = parser.parse_args()
    log.logger.info("the app args: %s" % args)

    app.run(host=args.host, port=args.port)

