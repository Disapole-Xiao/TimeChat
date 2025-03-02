# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# export CUDA_EMPTY_CACHE_PERCENT=80

export CUDA_VISIBLE_DEVICES=1,4

# python train.py --cfg-path  train_configs/test.yaml

# torchrun --nproc_per_node=2 --master_port 29501 train.py --cfg-path train_configs/tvg.yaml
torchrun --nproc_per_node=2 --master_port 29502 train.py --cfg-path train_configs/tvg_token.yaml