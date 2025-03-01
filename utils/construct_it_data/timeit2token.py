import os
import json
import random
from copy import deepcopy
import re
from tqdm import tqdm
import cv2

def get_video_duration(video_path):
    ''' get video duration in seconds '''
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        video.release()
        return round(duration, 2)
    except Exception as e:
        print(f'{video_path} read failed: {e}')
        return None

cfg = {
    'charades': {
        'anno_path': 'data/TimeIT/data/temporal_video_grounding/charades/instruct_tvg_12.4k_charades_with_duration.json',
        # 'num_sample': 1000
    },
    'didemo': {
        'anno_path': 'data/TimeIT/data/temporal_video_grounding/didemo/instruct_tvg_33.0k_didemo_with_duration.json',
        # 'num_sample': 1000
    },
    'activitynet': {
        'anno_path': 'data/TimeIT/data/temporal_video_grounding/activitynet/instruct_tvg_33.8k_anet_with_duration.json',
        # 'num_sample': 1000
    },
    
}

random.seed(33) # set seed
video_root = 'data/'
max_time_token = 600
types = ['timeit', 'token'] # timeit or token

for dataset, config in cfg.items():
    anno_path = config['anno_path']
    num_sample = config.get('num_sample', None)
    
    anno = json.load(open(anno_path, 'r'))
   
    new_anno_timeit = []
    new_anno_token = []
    
    for i in tqdm(range(num_sample if num_sample else len(anno)), desc=dataset):
        if num_sample:
            sample = random.choice(anno) # random sample
        else:
            sample = anno[i]

        if 'timeit' in types: 
            new_anno_timeit.append(deepcopy(sample))
        if 'token' in types: 
            # timestamp to token
            try:
                duration = sample['duration']
            except:
                print(f'no duration:\n{sample}')
                continue
            answer_pattern = r'The given query happens in (.*?) - (.*?) seconds'
            for qa in sample['QA']:
                start, end = re.match(answer_pattern, qa['a']).groups()
                start, end = float(start), float(end)
                start_token = f'<{round(start / duration * max_time_token)}>' # TODO 改为模型算损失前动态计算 target
                end_token = f'<{round(end / duration * max_time_token)}>'
                answer = f'The given query happens in <time>{start_token}{end_token}</time>.'
                qa['a'] = answer
            
            new_anno_token.append(sample)
    
    num_anno = len(new_anno_token)
    new_anno_token_path = f'data/token{max_time_token}/{dataset}_{num_anno}_token{max_time_token}.json'
    new_anno_timeit_path = f'data/tvg/{dataset}_{num_anno}.json'
    if 'timeit' in types:
        os.makedirs(f'data/token{max_time_token}', exist_ok=True)
        json.dump(new_anno_token, open(new_anno_token_path, 'w'))
        print('save to', new_anno_token_path)
    if 'token' in types:
        os.makedirs('data/tvg', exist_ok=True)
        json.dump(new_anno_timeit, open(new_anno_timeit_path, 'w'))
        print('save to', new_anno_timeit_path)

