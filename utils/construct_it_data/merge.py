import json
import random


def read_json(path):
    with open(path, "r") as fin:
        data = json.load(fin)
    return data

random.seed(33)

file_to_merge = [
    'data/tvg/activitynet_1000.json',
    'data/tvg/charades_1000.json',
    'data/tvg/didemo_1000.json'
]

# file_to_merge = [
#     'data/token600/activitynet_1000_token600.json',
#     'data/token600/charades_1000_token600.json',
#     'data/token600/didemo_1000_token600.json'
# ]

merge_data = []
for fi, fpath in enumerate(file_to_merge):
    data = read_json(fpath)
    for i, jterm in enumerate(data):
        data[i]["source"] = file_to_merge[fi].split("/")[-2]
    merge_data.extend(data)
    
random.shuffle(merge_data)

out_path = "data/tvg/instruct_time-sensitive_{}.json".format(round(len(merge_data)), 1)
# out_path = "data/token600/instruct_time-sensitive_{}_token600.json".format(round(len(merge_data)), 1)

print("save merge data at {}".format(out_path))
with open(out_path, "w") as fout:
    json.dump(merge_data, fout)
