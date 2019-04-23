#!/usr/bin/env sh
#
#$ -cwd
#$ -j y
#$ -N output_train_lstm
#$ -S /bin/sh
#

CUDA_VISIBLE_DEVICES=0 python main.py  --hdim=512 --output_height=256 --channels='32, 64, 128, 256, 512, 512' --m_plus=120 --weight_rec=0.05 --weight_kl=1.0  --weight_neg=0.5 --num_vae=0  --dataroot='/home/huaibo.huang/data/celeba-hq/celeba-hq-images-256' --trainsize=29000 --test_iter=1000 --save_iter=1 --start_epoch=0  --batchSize=16 --nrow=8 --lr_e=0.0002 --lr_g=0.0002   --cuda  --nEpochs=500   2>&1   | tee train0.log 

#Another appropriate setting defined as below:

#CUDA_VISIBLE_DEVICES=0 python main.py  --hdim=512 --output_height=256 --channels='32, 64, 128, 256, 512, 512' --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0  --weight_neg=0.5 --num_vae=10  --dataroot='/home/huaibo.huang/data/celeba-hq/celeba-hq-images-256' --trainsize=29000 --test_iter=1000 --save_iter=1 --start_epoch=0  --batchSize=16 --nrow=8 --lr_e=0.0002 --lr_g=0.0002   --cuda  --nEpochs=500   2>&1   | tee train0.log 
