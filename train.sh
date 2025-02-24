# torchrun --nproc_per_node=8 train.py --cfg-path  train_configs/stage2_finetune_time104k_valley72k.yaml
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
CUDA_EMPTY_CACHE_PERCENT=80
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path train_configs/tvg.yaml