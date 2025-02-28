import os
from tqdm import tqdm
import cv2
import json

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
    
video_dirs = ['data/Charades/videos', 'data/DiDeMo/videos', 'data/ActivityNet/train_videos', 'data/ActivityNet/val_videos', 'data/ActivityNet/test_videos']
for video_dir in video_dirs:
    res = {}
    video_files = os.listdir(video_dir)
    for video_file in tqdm(video_files, desc=video_dir, miniters=100):
        video_path = os.path.join(video_dir, video_file)
        duration = get_video_duration(video_path)
        if duration is None:
            print(f'{video_file} duration is None')
            continue
        res[video_file] = duration
    output_file = os.path.join(os.path.dirname(video_dir), 'video_durations.json')
    with open(output_file, '+') as f:
        old = json.load(f)
        old.update(res)
        json.dump(old, f)
    print(f'save {output_file}, {len(res)} durations / {len(video_files)} videos')
