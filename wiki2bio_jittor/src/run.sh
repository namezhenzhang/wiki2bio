# export JT_SYNC=1
# export trace_py_var=3

CUDA_VISIBLE_DEVICES=2,3,4

# nohup python wiki2bio_jittor/src/main.py \
# --mode test \
# --batch_size 64 \
# --seed 100 \
# --batch_size_valid 1 \
# --accumulation 1 \
# --learning_rate 0.0003 \
# --use_cuda True \
# --epoch 10 \
# --dual_attention True \
# --resume /home/zhangzhen/dir1/wiki2bio/wiki2bio_jittor/outputlogs/00026/res/checkpoints/9/model.ckpt \
# >dual_3.out 2>&1 &

# https://cg.cs.tsinghua.edu.cn/jittor/assets/cuda11.2_cudnn8_linux.tgz

python wiki2bio_jittor/src/main.py \
--mode test \
--batch_size 64 \
--seed 100 \
--batch_size_valid 128 \
--accumulation 1 \
--learning_rate 0.0003 \
--use_cuda True \
--epoch 10 \
--dual_attention True \
--resume /home/zhangzhen/dir1/wiki2bio/wiki2bio_jittor/outputlogs/00029/res/checkpoints/17/model.ckpt