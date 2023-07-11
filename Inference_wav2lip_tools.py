import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg
from config import log, params

# import sys
# print(sys.path)

# wav2lip包中
import audio
import face_detection
from models import Wav2Lip

'''
    获取视频帧序列
'''
def get_video_frame_list(face_video_path):
    video_stream = cv2.VideoCapture(face_video_path)
    fps = int(video_stream.get(cv2.CAP_PROP_FPS))

    log.logger.info('Reading video frames...')

    full_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break

        if params['app']['wav2lip']['resize_factor'] > 1:
            frame = cv2.resize(frame, (frame.shape[1] // params['app']['wav2lip']['resize_factor'], frame.shape[0] // params['app']['wav2lip']['resize_factor']))

        if params['app']['wav2lip']['rotate']:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

        y1, y2, x1, x2 = params['app']['wav2lip']['crop']
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)
    return full_frames, fps


'''
    获取音频帧序列
    :param audio_path 音频路径
    :param fps 按视频的fps做音频的
'''
def get_audio_frame_list(audio_path, fps):
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + params['app']['mel_step_size'] > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - params['app']['mel_step_size']:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + params['app']['mel_step_size']])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))
    return mel_chunks


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=cfg.device)
    batch_size = params['app']['wav2lip']['face_det_batch_size']

    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = params['app']['wav2lip']['pads']
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not params['app']['wav2lip']['nosmooth']: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

'''
    数据生成器，类似于torch.utils.data.Dataset
'''
def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if params['app']['wav2lip']['box'][0] == -1:
        if not params['app']['wav2lip']['static']:
            face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        log.logger.info('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = params['app']['wav2lip']['box']
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if params['app']['wav2lip']['static'] else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (params['app']['wav2lip']['img_size'], params['app']['wav2lip']['img_size']))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= params['app']['wav2lip']['wav2lip_batch_size']:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, params['app']['wav2lip']['img_size'] // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch    # face_batch, mel_batch, full_img_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, params['app']['wav2lip']['img_size'] // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

'''
    获取平滑之后的boxes
'''
def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def _load(checkpoint_path):
	if cfg.device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(cfg.device)
	return model.eval()



class Wav2lip_Dataset(Dataset):
    def __init__(self, frames, mels, resource_path):
        self.frames = frames
        self.mels = mels
        
        self.img_batch, self.mel_batch, self.frame_batch, self.coords_batch = [], [], [], []

        if params['app']['wav2lip']['box'][0] == -1:
            if not params['app']['wav2lip']['static']:
                if params['app']['wav2lip']['local_pose']:
                    # 如果脸部位置使用资源包，就从文件中读取list
                    log.logger.info("使用本地资源包 skipping face_detection, loading face coords from: %s" % (str(resource_path)))
                    with open(str(resource_path), "rb") as f:
                        self.face_det_results = pickle.load(f)
                else:
                    # self.face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
                    self.face_det_results = face_detect_by_codeformer(frames) # 将codeformer的pkl转为wav2lip能用的资源包
            else:
                self.face_det_results = face_detect([frames[0]])
        else:
            log.logger.info('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = params['app']['wav2lip']['box']
            self.face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]      
        
        
    def __len__(self):
        return len(self.frames)
    
    

    def __getitem__(self, index):
        idx = 0 if params['app']['wav2lip']['static'] else index % len(self.frames)
        
        m = self.mels[idx] # TODO: 确认mel这样取值是否无误 [80, 16]
        
        frame_to_save = self.frames[idx].copy()
        face, coords = self.face_det_results[idx].copy()

        face = cv2.resize(face, (params['app']['wav2lip']['img_size'], params['app']['wav2lip']['img_size']))
        
        return face, m, frame_to_save, coords






