# export JT_SYNC=1
# export trace_py_var=3

CUDA_VISIBLE_DEVICES=1

python wiki2bio_jittor/src/main.py \
--mode train \
--batch_size 64 \
--batch_size_valid 512 \
--accumulation 1 \
--use_cuda True \
--epoch 20 \
> output.txt

# https://cg.cs.tsinghua.edu.cn/jittor/assets/cuda11.2_cudnn8_linux.tgz