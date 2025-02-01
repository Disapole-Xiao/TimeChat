#!/bin/bash

DIR="results"
MODEL_DIR=${DIR}/timechat_7b.pth

# TASK='dvc'
# ANNO_DIR='data/TimeIT/data/dense_video_captioning/youcook2'
# VIDEO_DIR='data/YouCook2-BB/YouCook2_asr_denseCap/youcook2_6fps_224/'
# DATASET='youcook'
# SPLIT='val'
# PROMPT_FILE="prompts/${TASK}_description_zeroshot.txt"
# GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"
# ASR_DIR='data/YouCook2-BB/YouCook2_asr_denseCap/whisper_outputs_with_time/small.en.cleaned/'

#TASK='dvc'
#ANNO_DIR='data/TimeIT/data/dense_video_captioning/anet'
#VIDEO_DIR='data/Activitynet_Captions/anet_6fps_224/'
#DATASET='activitynet'
#SPLIT='test'
#PROMPT_FILE="prompts/${TASK}_description.txt"
#GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"

TASK='tvg'
ANNO_DIR='data/TimeIT/data/temporal_video_grounding/charades/charades_annotation'
VIDEO_DIR='data/Charades_v1_480/'
DATASET='charades'
SPLIT='test'
PROMPT_FILE="prompts/${TASK}_description_zeroshot.txt"
GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"
ASR_DIR='data/Charades/whisper_outputs_with_time/tiny.en.cleaned/'

#TASK='vhd'
#ANNO_DIR='data/TimeIT/data/video_highlight_detection/qvhighlights/annotations_raw'
#VIDEO_DIR='data/QVhighlights/videos/val/'
#DATASET='qvhighlights'
#SPLIT='val'
#PROMPT_FILE="prompts/${TASK}_description.txt"
#GT_FILE="${ANNO_DIR}/highlight_${SPLIT}_release.jsonl"
#ASR_DIR='data/QVhighlights/whisper_outputs_with_time/tiny.en.cleaned/val/'

NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}/${DATASET}
GPU_ID=3

python evaluate.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} \
--task ${TASK} --dataset ${DATASET} --output_dir ${OUTPUT_DIR} --split ${SPLIT} \
--num_frames ${NUM_FRAME} --batch_size 4 \
--prompt_file ${PROMPT_FILE} --timechat_model_path ${MODEL_DIR} \
--gpu_id ${GPU_ID} \
#--asr --asr_path ${ASR_DIR}
#--debug

python metrics/${TASK}/eval_${TASK}.py \
--pred_file "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json" \
--gt_file ${GT_FILE} \
--detail_file "${OUTPUT_DIR}/detail_${DATASET}_${SPLIT}_f${NUM_FRAME}.csv" \
| tee "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.txt"