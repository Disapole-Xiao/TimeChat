python utils/compress_video_data.py \
--input_root=../datasets/ActivityNet/videos/trn \
--output_root=data/ActivityNet/anet_6fps_224 \
--input_file_list_path=data/ActivityNet/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 24