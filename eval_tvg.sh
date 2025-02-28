#!/bin/bash

# for all tvg eval:
CKPT=timechat/ckpt/timechat/train_tvg_anet1k_charades1k_didemo1k_token600/20250227072/checkpoint_0.pth
# CKPT=timechat/ckpt/timechat/train_tvg_anet1k_charades1k_didemo1k/20250227090/checkpoint_0.pth
PROMPT_FILE="prompts/tvg_description.txt"
NUM_FRAME=16
BATCH_SIZE=4
OUTPUT_BASE=results/tvg
TOKEN=600
GPU_ID=1

DATASET=charades # charades, acitvitynet or didemo
SPLIT=test

OUTPUT_DIR=eval_results/tvg/token${TOKEN}_f${NUM_FRAME}_${DATASET}_${SPLIT}
python evaluate.py \
--dataset ${DATASET} \
--split ${SPLIT} \
--prompt_file ${PROMPT_FILE} \
--max_token ${TOKEN} \
--output_dir ${OUTPUT_DIR} \
--num_frames ${NUM_FRAME} \
--batch_size ${BATCH_SIZE} \
--timechat_model_path ${CKPT} \
--gpu_id ${GPU_ID} \
# --sample_num 500 \
# --debug

python metrics/tvg/eval_tvg.py --sample \
--pred_file ${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json \
--gt_file data/TimeIT/data/temporal_video_grounding/charades/charades_annotation/${SPLIT}.caption_coco_format.json \
> ${OUTPUT_DIR}/iou.txt
