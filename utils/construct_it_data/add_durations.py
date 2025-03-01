'''给 TimeIT 的已有标注加上 duration 属性。并过滤没有时长的数据'''

import json

def add_durations(data_path, duration_file, output_path):
    res = []
    with open(duration_file, 'r') as f:
        durations = json.load(f)
    with open(data_path, 'r') as f:
        data = json.load(f)
    for item in data:
        vid = item['video'].split('/')[-1]
        if vid in durations:
            item['duration'] = durations[vid]
            res.append(item)
        else:
            print(f'No duration found for {vid}')
    with open(output_path, 'w') as f:
        json.dump(res, f)
    print(f'Finished saving {len(res)} items to {output_path}')

if __name__ == '__main__':
    add_durations('data/TimeIT/data/temporal_video_grounding/activitynet/instruct_tvg_33.8k_anet.json',
                  'data/ActivityNet/video_durations.json',
                  'data/TimeIT/data/temporal_video_grounding/activitynet/instruct_tvg_33.8k_anet_with_duration.json')
    add_durations('data/TimeIT/data/temporal_video_grounding/charades/instruct_tvg_12.4k_charades.json',
                  'data/Charades/video_durations.json',
                    'data/TimeIT/data/temporal_video_grounding/charades/instruct_tvg_12.4k_charades_with_duration.json')
    add_durations('data/TimeIT/data/temporal_video_grounding/didemo/instruct_tvg_33.0k_didemo.json',
                    'data/DiDeMo/video_durations.json',
                    'data/TimeIT/data/temporal_video_grounding/didemo/instruct_tvg_33.0k_didemo_with_duration.json')