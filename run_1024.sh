#!/usr/bin/env sh
#
#$ -cwd
#$ -j y
#$ -N output_train_lstm
#$ -S /bin/sh
#

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  --hdim=512 --output_height=1024 --channels='16,32,64, 128, 256, 512, 512, 512' --m_plus=500 --weight_rec=0.01 --weight_kl=1.0  --weight_neg=1.0 --num_vae=4  --dataroot='/home/huaibo.huang/data/celeba-hq/celeba-hq-images' --trainsize=29000 --test_iter=1000 --save_iter=1 --start_epoch=0  --batchSize=12 --nrow=12 --lr_e=0.0002 --lr_g=0.0002   --cuda  --nEpochs=500  2>&1   | tee train0.log 

# --pretrained='model/net_model_epoch_70_iter_0.pth'
