import os
import json
import random
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
        'anno_path': 'data/TimeIT/data/temporal_video_grounding/charades/instruct_tvg_12.4k_charades.json',
        # 'duration_file': 'data/Charades/video_durations.csv',
        'num_sample': 10
    },
    'didemo': {
        'anno_path': 'data/TimeIT/data/temporal_video_grounding/didemo/instruct_tvg_33.0k_didemo.json',
        # 'duration_file': 'data/DiDeMo/video_durations.json',
        'num_sample': 10
    },
    'activitynet': {
        'anno_path': 'data/TimeIT/data/temporal_video_grounding/activitynet/instruct_tvg_33.8k_anet.json',
        # 'duration_file': 'data/ActivityNet/video_durations.json',
        'num_sample': 10
    }
}

random.seed(33) # set seed
video_root = 'data/'
max_time_token = 600
types = ['timeit', 'token'] # timeit or token

for dataset, config in cfg.items():
    anno_path = config['anno_path']
    duration_file = config.get('duration_file', '')
    num_sample = config['num_sample']
    
    anno = json.load(open(anno_path, 'r'))
    new_anno_token_path = f'data/token{max_time_token}/{dataset}_{num_sample}_token{max_time_token}.json'
    new_anno_timeit_path = f'data/tvg/{dataset}_{num_sample}.json'
    new_anno_token = []
    new_anno_timeit = []
    durations = {}

    # load duration file
    if duration_file.endswith('.json'):
        durations = json.load(open(duration_file, 'r'))
    elif duration_file.endswith('.csv'):
        with open(duration_file, 'r') as f:
            for line in f:
                v_id, sec = line.strip().split(',')
                durations[v_id] = float(sec)
    
    for _ in tqdm(range(num_sample), desc=dataset):
        for try_i in range(5): # max try
            # random sample
            sample = random.choice(anno)
            video = sample['video'] # TimeIT video 形式：Charades/videos/AO8RW.mp4
            # if video exists
            vid_path = os.path.join(video_root, video)
            if not os.path.exists(vid_path):
                print(f'{vid_path} not exists, retry {try_i}')
                continue    
            # load duration, if not exists, get duration or retry
            if video not in durations:
                print(f'{video}: opencv read duration...')
                durations[video] = get_video_duration(vid_path)
            if durations[video] is None:
                print(f'{video} duration not found or is None, retry {try_i}')
                continue
            if 'timeit' in types: 
                new_anno_timeit.append(sample)
            if 'token' not in types: 
                break
            duration = durations[video]
            # timestamp to token
            answer_pattern = r'The given query happens in (.*?) - (.*?) seconds'
            start, end = re.match(answer_pattern, sample['QA'][0]['a']).groups()
            start, end = float(start), float(end)
            start_token = f'<{round(start / duration * max_time_token)}>' # TODO 改为模型算损失前动态计算 target
            end_token = f'<{round(end / duration * max_time_token)}>'
            answer = f'The given query happens in <time>{start_token}{end_token}</time>.'

            new_anno_token.append({
                'video': video,
                'QA': [{
                    'q': sample['QA'][0]['q'],
                    'a': answer
                }],
                'duration': duration
            })
            break

        if try_i == 4:
            raise ValueError('retry too many times')
        
    if 'timeit' in types:
        os.makedirs(f'data/token{max_time_token}', exist_ok=True)
        json.dump(new_anno_token, open(new_anno_token_path, 'w'))
        print('save to', new_anno_token_path)
    if 'token' in types:
        os.makedirs('data/tvg', exist_ok=True)
        json.dump(new_anno_timeit, open(new_anno_timeit_path, 'w'))
        print('save to', new_anno_timeit_path)

