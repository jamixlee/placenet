#!/bin/sh

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.1 python train.py \
  --num_epoch=10 \
  --num_batch=36 \
  --num_layer=12 \
  --obs_range=20 \
  --obs_count=5 \
  --obs_train \
  --attention=rdo \
  --att_weight=0.1 \
  --att_weight_grad \
  --att_weight_delay \
  --data_dir=/data/house-v3 \
  --data_dir_test=test_seen \
  --root_log_dir=../logs \
  --log_dir=Test_01 \
  --log_interval=100 \
  --num_saved_model=2 \
  --model_save_interval=3 \
  --workers=8 \
  --gpu=0
