#!/bin/bash
# the consistency weight for each pyramid is decreas
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot   /home/imre/hebin/wash_ink_data_set/horse2  --name horse2_cyclegan_edge_1_dec_150 --model cycle_gan --no_dropout --display_port 8103 --display_freq 10 --lambda_sup 10 --lambda_A 10 --lambda_B 10 --start_dec_sup 150 --lambda_ink 0.05 --loadSize 286 --fineSize 256
