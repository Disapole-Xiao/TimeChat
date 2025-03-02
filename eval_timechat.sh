#!/bin/bash

# for all tvg eval:
PROMPT_FILE="prompts/tvg_description_zeroshot.txt"
TIME=$(date +"%m%d%H%M")
NUM_FRAME=96
BATCH_SIZE=4

CKPT=ckpt/timechat/timechat_7b_paper.pth

TOKEN=0
GPU_ID=3

DATASET=charades # charades, activitynet or didemo
SPLIT=test # train, val or test

OUTPUT_DIR=results/tvg/timechat_f${NUM_FRAME}_${DATASET}_${SPLIT}_${TIME}
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