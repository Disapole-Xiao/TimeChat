ls -U ../datasets/ActivityNet/videos/trn/ >> data/ActivityNet/train_names.txt
python utils/compress_video_data.py \
--input_root=../datasets/ActivityNet/videos/trn \
--output_root=data/ActivityNet/train_videos \
--input_file_list_path=data/ActivityNet/train_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 6

ls -U ../datasets/ActivityNet/videos/val/ >> data/ActivityNet/val_names.txt
python utils/compress_video_data.py \
--input_root=../datasets/ActivityNet/videos/val \
--output_root=data/ActivityNet/val_videos \
--input_file_list_path=data/ActivityNet/val_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 6

ls -U ../datasets/ActivityNet/videos/tst/ >> data/ActivityNet/test_names.txt
python utils/compress_video_data.py \
--input_root=../datasets/ActivityNet/videos/tst \
--output_root=data/ActivityNet/test_videos \
--input_file_list_path=data/ActivityNet/test_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 6