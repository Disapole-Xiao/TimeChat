import json
import random


def read_json(path):
    with open(path, "r") as fin:
        data = json.load(fin)
    return data


file_to_merge = [
    'data/tvg/activitynet_10.json',
    'data/tvg/charades_10.json',
    'data/tvg/didemo_10.json'
]

merge_data = []
for fi, fpath in enumerate(file_to_merge):
    data = read_json(fpath)
    for i, jterm in enumerate(data):
        data[i]["source"] = file_to_merge[fi].split("/")[-2]
    merge_data.extend(data)
    
random.shuffle(merge_data)

out_path = "data/tvg/instruct_time-sensitive_{}.json".format(round(len(merge_data)), 1)
print("save merge data at {}".format(out_path))
with open(out_path, "w") as fout:
    json.dump(merge_data, fout)
