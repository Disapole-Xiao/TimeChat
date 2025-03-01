import json
from pathlib import Path
import time
import datetime
import copy
import os
import random
from math import ceil
import logging

import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from timechat.common.logger import setup_logger
from timechat.common.config import Config
from timechat.common.registry import registry
from timechat.conversation.conversation_video_batch import Chat, default_conversation, conv_llava_llama_2
from utils.format_dvc import format_dvc_output
from utils.format_tvg import format_tvg_output
from utils.format_vhd import format_vhd_output

def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args, anno_path, split=None):
    """从 JSON 文件中加载 coco 格式注释数据。
    如果 `args.debug` 为 `True`，则只返回前10个注释。

    anno data example:
    ```
        "annotations": [
            {
                "image_id": "3MSZA.mp4",
                "caption": "person turn a light on.",
                "timestamp": [
                    24.3,
                    30.4
                ],
                "id": 0
            },
            ...
        ]
    ```
    """
    file_path = os.path.join(anno_path, f'{split}.caption_coco_format.json')
    with open(file_path, 'r') as f:
        data = json.load(f)["annotations"]

    if args.debug:
        data = data[:10]
    return data

def save_result(args, output_dir, results, split_name='test', format=False):
    """保存结果到文件。

    Args:
        results: 数据
        format: 是否添加 `fmt` 前缀
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result.json'
    if args.debug:
        file_name = 'debug_' + file_name
    if format:
        file_name = 'fmt_' + file_name
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f, indent=4)
    return


def format_dvc(datas):
    fmt_datas = {}
    timestamp_count = []
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        caption = jterm["generated_cap"]
        timestamps, sents = format_dvc_output(caption)
        if len(timestamps) == 0:
            cnt += 1
            print(vid, caption)
        fmt_datas[vid] = []
        for j in range(len(timestamps)):
            fmt_datas[vid].append({"timestamp": timestamps[j], "caption": sents[j]})
        timestamp_count.append(len(timestamps))
    print(f"predict avg {sum(timestamp_count) / len(timestamp_count)} events per video")
    print(f'parse failed number: {cnt}')
    return fmt_datas


def format_tvg(datas, max_token=None):
    """提取时间戳。对于 time_token，转化为时间戳
    output:
    ```
        "4": {
            "vname": "AMT7R.mp4"
            "query": "a person is putting a picture onto the wall.",
            "timestamp": [
                [
                    0.0,
                    29.7
                ]
            ],
        },
    ```
    """
    fmt_datas = {}
    cnt = 0
    for jterm in datas:
        vname = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        anno_id = int(jterm["id"])
        duration = jterm["duration"]
        timestamps = format_tvg_output(gcap, args.max_token, duration)
        if len(timestamps) == 0: # 未提取到时间戳
            cnt += 1
            print(f'Fail to extract timestamps: {vname}, {query}\n\t{gcap}\n')
        fmt_datas[anno_id] = {"vname": vname, "query": query, "timestamp": timestamps, "duration": duration}
    print(f'parse failed number: {cnt}')
    return fmt_datas


def format_vhd(datas, gts):
    vid2gts = {}
    for jterm in gts:
        vid2gts[jterm["image_id"]] = jterm
    fmt_datas = []
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        qid = jterm["id"]
        highlights, clipscores = format_vhd_output(gcap, vid2gts[vid])
        if len(highlights) == 0:
            cnt += 1
            print(vid, query + "\n", gcap + "\n")
            # pdb.set_trace()
        else:
            # print(gcap)
            # print(timestamps)
            pass
        result = {}
        result["qid"] = qid
        result["query"] = query
        result["vid"] = vid
        result["pred_saliency_scores"] = clipscores
        fmt_datas.append(result)
    print(f'parse failed number: {cnt}')
    return fmt_datas


def generate(chat, gr_videos, user_messages, num_beams, temperature, top_p, n_frms, chat_states=None, img_lists=None):
    """模型生成结果

    Returns:
        tuple: `(responses, chat_states, img_lists)`
    """
    N = len(user_messages)
    if chat_states is None:
        chat_states = []
        for i in range(N):
            if args.model_type == 'vicuna':
                chat_state = default_conversation.copy()
            else:
                chat_state = conv_llava_llama_2.copy()
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            chat_states.append(chat_state)
    if img_lists is None:
        img_lists = [[] for i in range(N)]
        chat.upload_video_without_audio(gr_videos, chat_states, img_lists, n_frms=n_frms)

    for user_message, chat_state in zip(user_messages, chat_states):
        chat.ask(user_message, chat_state)

    responses = chat.answer(convs=chat_states,
                            img_lists=img_lists,
                            num_beams=num_beams,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=512,
                            max_length=3000)[0]
    return responses, chat_states, img_lists


def main(args):
    num_beams = 1
    temperature = args.temperature
    top_p = args.top_p
    n_frms = args.num_frames
    eval_start_time = time.time()
    prompt = read_txt(args.prompt_file)

    # load model
    device = torch.device(f"cuda:{args.gpu_id}")
    args.options = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = args.timechat_model_path
    if args.no_lora:
        model_config.lora = False

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()
    message = '\n' + '\n'.join([f'{k:<25}: {v}' for k, v in vars(args).items()])
    logging.info(message)

    # init model
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.time_instruct.vis_processor.eval
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device=device)
    print('Initialization Finished')

    # get anno/video path
    if args.dataset == 'charades':
        anno_path = f'data/TimeIT/data/temporal_video_grounding/charades/charades_annotation/'
        video_path = 'data/Charades/videos'
    elif args.dataset == 'activitynet':
        anno_path = f'data/TimeIT/data/temporal_video_grounding/activitynet/activitynet_annotation/'
        video_path = f'data/ActivityNet/anet_6fps_224'
    # elif args.dataset == 'didemo':
    #     anno_path = f'data/TimeIT/data/temporal_video_grounding/didemo/didemo_annotation/'
    #     video_path = 'data/DiDeMo/videos'
    
    # 如果传入了路径，直接使用
    if args.anno_path: anno_path = args.anno_path
    if args.video_path: video_path = args.video_path

    assert anno_path is not None and video_path is not None, "The dataset is not supported, please provide the your own ann_path and video_path"
    
    # load data
    anno_data = load_data(args, anno_path, split=args.split)
    vpaths = []
    vnames = [] 
    queries = []
    anno_ids = []
    vdurations = []
    if args.sample_num > 0:
        # sample part data to evaluate
        if (args.sample_num > len(anno_data)):
            print(f"Sample number {args.sample_num} is larger than the total number of data {len(anno_data)}, use {len(anno_data)} data")
        else:
            anno_data = random.sample(anno_data, args.sample_num)
    for jterm in anno_data:
        vname = jterm["image_id"].split("/")[-1]
        vid_path = os.path.join(video_path, vname)
        vpaths.append(vid_path)
        vnames.append(vname)
        queries.append(jterm["caption"])
        anno_ids.append(jterm["id"])
        vdurations.append(jterm["duration"])

    # evaluate using batch
    results = []
    bz = args.batch_size
    iter = ceil(len(vnames) / bz)
    for i in tqdm(range(iter)):
        sid = i * bz
        eid = min((i + 1) * bz, len(vnames))
        prompts = []
        # load video
        paths = vpaths[sid:eid]
        for pi in range(len(paths)):
            final_prompt = copy.deepcopy(prompt)
            if args.task in ["tvg", "vhd"]:
                idx = sid + pi
                prompts.append(final_prompt.format(args.dataset, queries[idx].strip('.')))
            else: # dvc
                prompts.append(final_prompt)
        outputs, chat_states, img_lists = generate(chat, paths, prompts, num_beams, temperature, top_p, n_frms)
        if args.post_check: # 让模型再检查一遍生成格式
            post_check_prompt = read_txt(args.post_check_prompt_file)
            post_check_prompts = [post_check_prompt] * len(paths)
            outputs, chat_states, img_lists = generate(chat, paths, post_check_prompts, num_beams, temperature, top_p,
                                                       n_frms, chat_states, img_lists)
        for j, (output, chat_state) in enumerate(zip(outputs, chat_states)):
            if args.task in ["tvg", "vhd"]:
                result = {
                    "id": anno_ids[sid + j],
                    "vname": vnames[sid + j],
                    "query": queries[sid + j],
                    "generated_cap": output,
                    "prompt": chat_state.get_prompt(),
                    "duration": vdurations[sid + j]
                }
            else:
                result = {
                    "vname": vnames[sid + j],
                    "prompt": chat_state.get_prompt(),
                    "generated_cap": output,
                    "duration": vdurations[sid + j]
                }
            results.append(result)
            # 前 5 个 iter 输出结果
            if i < 5: 
                print(*[f'{k}: {v}' for k, v in result.items()], sep='\n')
                print('*' * 50)

    # save results
    save_result(args, args.output_dir, results, args.split)

    # format results to calculate metrics
    if args.task == "dvc":
        fmt_results = format_dvc(results)
    elif args.task == "tvg":
        fmt_results = format_tvg(results, args.max_token)
    elif args.task == "vhd":
        fmt_results = format_vhd(results, anno_data)
    else:
        print(f"Not support formatting samples for task {args.task}")
    # save format results
    save_result(args, args.output_dir, fmt_results, args.split, format=True)

    # evaluate time
    total_time = time.time() - eval_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time))) # convert seconds to date
    print('Evaluate time {}'.format(total_time_str))

    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(json.dumps(cfg.to_dict(), indent=4) + "\n")
        f.write(message + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='eval_configs/timechat.yaml')
    parser.add_argument('--model_type', choices=['llama_v2', 'vicuna'], default='llama_v2')
    parser.add_argument('--timechat_model_path', default='ckpt/timechat/timechat_7b.pth')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--task', choices=['tvg', 'dvc', 'vhd'], default='tvg')
    parser.add_argument('--dataset', default='charades', help='charades, activitynet, didemo')
    parser.add_argument('--split', default='test')
    parser.add_argument('--anno_path', type=str, default=None)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--sample_num', type=int, default=-1, help='fast inference by sampling N instances to evaluate')
    parser.add_argument('--prompt_file', default='prompts/tvg_description.txt')
    parser.add_argument('--post_check', action='store_true', help='post check the format of generated captions')
    parser.add_argument('--post_check_prompt_file', type=str, default='prompts/tvg_post_check.txt')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--max_token', type=int, default=None, help='最大的 time_token 号')

    # parser.add_argument('--timestamp', action='store_true', help='input the gt/predicted timestamps to the model')
    # parser.add_argument('--timestamp_file', type=str, default='', help='the predicted timestamps file')
    # parser.add_argument('--asr', action='store_true')
    # parser.add_argument('--asr_path', type=str, default='data/YouCook2-BB/YouCook2_asr_denseCap/whisper_outputs_with_time/small.en.cleaned/')

    args = parser.parse_args()
    main(args)
