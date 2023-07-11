import os
import yaml
import torch

from common.Logger import Logger as Logger


logfile = "./log/"
os.makedirs(logfile, exist_ok=True)
log = Logger(os.path.join(logfile, "cvcuda_facestorehelper.log"), level='info')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log.logger.info("device: {}".format(device))

with open("./cvcuda_facestorehelper.yaml", 'r', encoding='utf-8') as f:
    params = yaml.load(stream=f, Loader=yaml.FullLoader)

os.makedirs(params['app']['audio_input_path'], exist_ok=True)
os.makedirs(params['app']['video_based_path'], exist_ok=True)
os.makedirs(params['app']['final_output_path'], exist_ok=True)
