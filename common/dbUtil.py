import os
import sys
curr_path = os.path.abspath(os.path.dirname(__file__))
proj_path = curr_path[:curr_path.find('common')]
sys.path.append(proj_path)
import re
import yaml
import pymysql
import datetime
import traceback


from config import params
from config import log

# with open("../talking_head.yaml", 'r', encoding='utf-8') as f:
#     params = yaml.load(stream=f, Loader=yaml.FullLoader)

'''
    数据库操作工具
'''

'''
    数据库连接类
'''
class DB_Connection():
    def __init__(self):
        self.conn = None
        self.cursor = None

    def begin(self):
        self.conn = pymysql.connect(host=params['db']['db_ip'],
                                    port=params['db']['db_port'],
                                    user=params['db']['db_username'],
                                    password=params['db']['db_password'],
                                    database=params['db']['db_dbname'],
                                    charset=params['db']['db_charset'])
        self.conn.begin()
        self.cursor = self.conn.cursor()

    def close(self):
        self.conn.close()


db_connection = DB_Connection()


#
# conn = pymysql.connect(host=params['db']['db_ip'],
#                        port=params['db']['db_port'],
#                        user=params['db']['db_username'],
#                        password=params['db']['db_password'],
#                        database=params['db']['db_dbname'],
#                        charset=params['db']['db_charset'])
# cursor = conn.cursor()
#
# conn_main = pymysql.connect(host=params['db']['db_ip'],
#                        port=params['db']['db_port'],
#                        user=params['db']['db_username'],
#                        password=params['db']['db_password'],
#                        database=params['db']['db_dbname'],
#                        charset=params['db']['db_charset'])
# cursor_main = conn.cursor()



def get_virtual_resources(VirtualmanKey):
    '''
        根据虚拟人ID查找资源包
    :param VirtualmanKey: 虚拟人ID
    :return: 返回资源包的路径；
    '''
    if table_exists("tab_virtual_man_info") is False:
        create_tab_virtual_man_info()
        return None

    sql = '''
        select VirtualmanKey, resources_path, enable_datetime, term_of_validity, is_enabled from tab_virtual_man_info where is_enabled='1' and VirtualmanKey=%s;
    ''' % (VirtualmanKey)

    db_connection.begin()

    db_connection.cursor.execute(sql)
    results = db_connection.cursor.fetchall()
    virtual_resources = results[0][1]

    db_connection.close()

    return virtual_resources


def save_progress(taskID, start_time, curr_time, status, info, progress):
    if table_exists("tab_progress") is False:
        create_tab_progress()

    try:
        sql = '''
                insert into tab_progress (taskID, start_time, curr_time, status, info, progress)
                VALUES ('%s', '%s', '%s', '%s', '%s', '%d')
            ''' % (taskID, start_time, curr_time, status, info, progress)

        db_connection.begin()

        db_connection.cursor.execute(sql)
        db_connection.conn.commit()

        db_connection.close()

        log.logger.info("save the task progress to db: %s" % sql)
    except Exception as e:
        log.logger.error(traceback.format_exc())
        log.logger.error("save the task progress to db failed: %s" % sql)
        db_connection.conn.rollback()

        db_connection.close()


def get_progress_by_taskid(taskID):
    '''
        根据taskID查询进度
    :param taskID: 任务ID
    :return: dict
    '''

    sql = '''
        select taskID, start_time, curr_time, status, info, progress from tab_progress 
        where taskID='%s' order by progress desc limit 1;
    ''' % (taskID)

    db_connection.begin()

    db_connection.cursor.execute(sql)
    results = db_connection.cursor.fetchall()

    ret_dict = {}
    ret_dict['taskID'] = taskID

    if len(results) == 0:
        ret_dict['start_time'] = None
        ret_dict['curr_time'] = None
        ret_dict['status'] = None
        ret_dict['info'] = None
        ret_dict['progress'] = 0
    else:
        row = results[0]
        ret_dict['start_time'] = row[1]
        ret_dict['curr_time'] = row[2]
        ret_dict['status'] = row[3]
        ret_dict['info'] = row[4]
        ret_dict['progress'] = row[5]

    db_connection.close()

    return ret_dict



def save_task_info(taskID, status, MediaURL, create_time, end_time, VirtualmanKey, content):
    if table_exists("tab_taskinfo") is False:
        create_tab_taskinfo()

    try:
        sql = '''
                insert into tab_taskinfo (taskID, status, MediaURL, create_time, end_time, VirtualmanKey, content)
                VALUES ("%s", "%s", "%s", "%s", "%s", "%s", "%s")
            ''' % (taskID, status, MediaURL, create_time, end_time, VirtualmanKey, content)

        db_connection.begin()

        db_connection.cursor.execute(sql)
        db_connection.conn.commit()

        db_connection.close()

        log.logger.info("save the task info to db: %s" % sql)
    except Exception as e:
        log.logger.error(traceback.format_exc())
        log.logger.error("save the task info to db failed: %s" % sql)
        db_connection.conn.rollback()
        db_connection.close()

'''
    更新任务信息
'''
def update_task_info(taskID, status, MediaURL, end_time):
    try:
        sql = '''
            update tab_taskinfo set status='%s', MediaURL='%s', end_time='%s' where taskID='%s'
        ''' % (status, MediaURL, end_time, taskID)

        db_connection.begin()

        db_connection.cursor.execute(sql)
        db_connection.conn.commit()

        db_connection.close()

        log.logger.info("任务信息已更新：%s" % sql)
    except Exception as e:
        log.logger.error(traceback.format_exc())
        log.logger.error("save the task info to db failed: %s" % sql)
        db_connection.conn.rollback()

        db_connection.close()


'''
    获取待办列表
'''
def get_task_todo_list():
    sql = '''
            select taskID, status, MediaURL, create_time, end_time, VirtualmanKey, content from tab_taskinfo 
            where status="-1" order by create_time asc;
        '''

    db_connection.begin()

    db_connection.conn.begin()
    db_connection.cursor.execute(sql)
    results = db_connection.cursor.fetchall()

    task_info_list = []
    for row in results:
        task_info_list.append((row[0], row[5], row[6]))
    db_connection.close()
    return task_info_list


'''
    通过taskID获取MediaURL
'''
def get_MediaURL_by_taskid(task_id):
    sql = '''
            select taskID, status, MediaURL, create_time, end_time, VirtualmanKey, content from tab_taskinfo 
            where taskID='%s';
        ''' % task_id

    db_connection.begin()

    db_connection.cursor.execute(sql)
    results = db_connection.cursor.fetchall()

    for row in results:
        print(row)
    mediaURL = results[0][2]
    db_connection.close()

    return mediaURL

'''
    判断表是否存在
    :param table_name 表名
    :return True，表存在；False，表不存在
'''
def table_exists(table_name):
    sql = "show tables;"

    db_connection.begin()

    db_connection.cursor.execute(sql)
    tables = [db_connection.cursor.fetchall()]
    table_list = re.findall('(\'.*?\')', str(tables))
    table_list = [re.sub("'", '', each) for each in table_list]

    db_connection.close()
    if table_name in table_list:
        return True
    else:
        return False


'''
    创建虚拟人信息表
'''
def create_tab_virtual_man_info():
    sql = '''
        CREATE TABLE `tab_virtual_man_info`  (
          `VirtualmanKey` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '虚拟人ID',
          `resources_path` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '虚拟人资源包',
          `enable_datetime` datetime(0) NULL DEFAULT NULL COMMENT '启用时间',
          `term_of_validity` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '有效期',
          `is_enabled` int(0) NULL DEFAULT NULL COMMENT '是否已启用：1是，0否；'
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 ROW_FORMAT = Dynamic;
    '''

    db_connection.begin()

    ret = db_connection.cursor.execute(sql)

    flag = ret
    db_connection.close()

    return flag


'''
    创建进度信息表
'''
def create_tab_progress():
    sql = '''
        CREATE TABLE `tab_progress`  (
          `taskID` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '任务ID',
          `start_time` datetime(0) NULL DEFAULT NULL COMMENT '任务的创建时间',
          `curr_time` datetime(0) NULL DEFAULT NULL COMMENT '当前时间',
          `status` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '当前状态：COMMIT已提交需要排队；MAKING制作中；SUCCESS制作成功；FAIL制作失败',
          `info` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '进度信息',
          `progress` float(5, 2) NULL DEFAULT NULL COMMENT '进度值'
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 ROW_FORMAT = Dynamic;
    '''

    db_connection.begin()

    ret = db_connection.cursor.execute(sql)

    flag = ret
    db_connection.close()

    return flag

'''
    创建任务信息表
'''
def create_tab_taskinfo():
    sql = '''
            CREATE TABLE `tab_taskinfo`  (
              `taskID` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '任务ID',
              `status` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '任务状态：1成功，0失败，-1待办',
              `MediaURL` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '给到工程侧的URL',
              `create_time` datetime(0) NULL DEFAULT NULL COMMENT '任务开始时间',
              `end_time` datetime(0) NULL DEFAULT NULL COMMENT '任务结束时间',
              `VirtualmanKey` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '虚拟人ID',
              `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '其他参数'
            ) ENGINE = InnoDB CHARACTER SET = utf8mb4 ROW_FORMAT = Dynamic;
        '''

    db_connection.begin()

    ret = db_connection.cursor.execute(sql)

    flag = ret
    db_connection.close()

    return flag


if __name__ == '__main__':
    # result = get_virtual_resources("123")
    # print(result)

    # save_progress(taskID='123_1684231888.9023688',
    #               create_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    #               curr_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    #               status='SUCCESS',
    #               info='视频生成成功',
    #               progress=100
    #               )

    # results = get_progress_by_taskid(taskID='123456')
    # print(results)

    content = '''
        {
           "Header": {},
           "Payload": {
               "VirtualmanKey": "123",
               "Audio_url": "http://tts.dui.ai/runtime/v1/cache/944333344422222291?productId=279615089&aispeech-da-env=hd-ack",
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

    # save_task_info(taskID='123456',
    #                status='-1',
    #                MediaURL='',
    #                create_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    #                end_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    #                VirtualmanKey='123',
    #                content=content)

    # results = get_task_todo_list()
    # for r in results:
    #     print(r)

    update_task_info('123456', '0', 'http://psyai.net/public/123456.mp4', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))



