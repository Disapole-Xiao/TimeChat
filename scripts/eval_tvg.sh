#!/bin/bash

# for all tvg eval:
PROMPT_FILE="prompts/tvg_description.txt" # prompts/tvg_description.txt or prompts/tvg_description_zeroshot.txt
TIME=$(date +"%m%d%H%M")
NUM_FRAME=16
BATCH_SIZE=8
CKPT=timechat/ckpt/timechat/train_tvg_anet_charades_didemo_full_fixqformer_f16_token600/20250302081/checkpoint_0.pth
# CKPT=timechat/ckpt/timechat/train_tvg_anet_charades_didemo_full_fixqformer_f16/20250302081/checkpoint_0.pth

TOKEN=600
GPU_ID=1

DATASET=charades # charades, activitynet or didemo
SPLIT=test # train, val or test

OUTPUT_DIR=results/tvg/full_token${TOKEN}_fixqformer_f${NUM_FRAME}_${DATASET}_${SPLIT}_${TIME}
python evaluate.py \
--dataset ${DATASET} \
--split ${SPLIT} \
--prompt_file ${PROMPT_FILE} \
--max_token ${TOKEN} \
--output_dir ${OUTPUT_DIR} \
--num_frames ${NUM_FRAME} \
--batch_size ${BATCH_SIZE} \
--timechat_model_path ${CKPT} \
--gpu_id ${GPU_ID} && \
python metrics/tvg/eval_tvg.py --sample \
--pred_file ${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json \
--gt_file data/TimeIT/data/temporal_video_grounding/${DATASET}/${DATASET}_annotation/${SPLIT}.caption_coco_format.json \
> ${OUTPUT_DIR}/iou.txt && \
python ~/send_email.py -t 'success '${OUTPUT_DIR} || \
python ~/send_email.py -t 'fail '${OUTPUT_DIR}