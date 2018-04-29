#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python main.py --is_train False --check_version 552000 --batch_size 50 --units 512 --symbols 20000 --inference_path weibo_pair_test_Q.post
