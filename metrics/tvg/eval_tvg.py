import json
import os
import time
import sys
import argparse
import pdb
import csv

def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    return max(min1 - max0, 0) / (max1 - min0)


def toSec(timeStr):
    t = time.strptime(timeStr, "%H:%M:%S")
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

def captiondata_modify(steps):
    modify_data = {}
    for i, step in enumerate(steps[0]):
        for key in step["step"].keys():
            name = step["step"][key]["query_idx"]
            modify_data[name] = [[step['step'][key]["startime"], step['step'][key]["endtime"]]]
        
    return modify_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="/home/yaolinli/code/Ask-Anything/video_chat/output/eval_7b_tvg_charades/fmt_charades_test_f8_result.json")
    parser.add_argument('--gt_file', type=str, default='/home/yaolinli/dataset/Charades/charades_annotation/test.caption_coco_format.json')
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--detail_file', type=str, default=None, help='输出详细评估结果的文件路径')
    args = parser.parse_args()
    '''
    {
        "query_idx": [start_time, end_time],
        ...
    }
    '''
    answer = read_json(args.gt_file)
    answer = answer["annotations"]
    gt_timestamps = {}
    for jterm in answer:
        gt_timestamps[jterm["id"]] = jterm["timestamp"]
        
    submission = read_json(args.pred_file)
    pred_timestamps = {}
    for qid, jterm in submission.items():
        pred_timestamps[int(qid)] = jterm["timestamp"]
    
    if args.sample:
        new = {}
        for qid in pred_timestamps.keys():
            new[qid] = gt_timestamps[qid]
        gt_timestamps = new
    num = len(gt_timestamps)
    print(f"# pred video timestamps {len(pred_timestamps)}; # gt video timestamps {len(gt_timestamps)}")
    assert len(gt_timestamps) == len(pred_timestamps)
    Result = {0.3:0, 0.5:0, 0.7:0}
    detailed_results = []

    for key in gt_timestamps.keys():
        iou_val = 0.0
        if len(pred_timestamps[key]) >= 1:
            iou_val = iou(gt_timestamps[key], pred_timestamps[key][0])
        for c_iou in Result.keys():
            if(iou_val >= c_iou):
                Result[c_iou] = Result[c_iou] + 1
        # 生成并保存到 CSV
        if args.detail_file:
            id = key
            vid = submission[str(key)]["vid"]
            query = submission[str(key)]["query"]
            gt_s, gt_e = gt_timestamps[key]
            out_s, out_e = pred_timestamps[key][0] if len(pred_timestamps[key])>=1 else [None, None] # 预测时间戳，没有则为空
            detailed_results.append([
                id, vid, query, iou_val, gt_s, gt_e, out_s, out_e
            ])

    # 输出召回率
    for key in Result.keys():
        print(f"IOU {key}: {Result[key]*100/num}")

    # 保存详细结果
    if args.detail_file:
        with open(args.detail_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'vid','query','iou','gt_s','gt_e','output_s','output_e'])
            writer.writerows(detailed_results)