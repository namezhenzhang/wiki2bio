# export JT_SYNC=1
# export trace_py_var=3

CUDA_VISIBLE_DEVICES=2,3,4

nohup python wiki2bio_jittor/src/main.py \
--mode train \
--batch_size 32 \
--seed 666 \
--batch_size_valid 512 \
--accumulation 1 \
--learning_rate 0.0005 \
--use_cuda True \
--epoch 20 \
--dual_attention False \
>no_dual.out 2>&1 &

# https://cg.cs.tsinghua.edu.cn/jittor/assets/cuda11.2_cudnn8_linux.tgz