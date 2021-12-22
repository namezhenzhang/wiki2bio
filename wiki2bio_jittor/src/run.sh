# export JT_SYNC=1
# export trace_py_var=3

CUDA_VISIBLE_DEVICES=1

python wiki2bio_jittor/src/main.py \
--mode test \
--batch_size 64 \
--batch_size_valid 256 \
--accumulation 1 \
--learning_rate 0.0005 \
--use_cuda True \
--epoch 8 \
# https://cg.cs.tsinghua.edu.cn/jittor/assets/cuda11.2_cudnn8_linux.tgz