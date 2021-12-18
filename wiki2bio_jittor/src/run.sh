# export JT_SYNC=1
# export trace_py_var=3
 
python wiki2bio_jittor/src/main.py \
--mode train \
--batch_size 64 \
--batch_size_valid 128 \
--accumulation 1 \
--use_cuda True \
--epoch 1 \
--report 1

# https://cg.cs.tsinghua.edu.cn/jittor/assets/cuda11.2_cudnn8_linux.tgz