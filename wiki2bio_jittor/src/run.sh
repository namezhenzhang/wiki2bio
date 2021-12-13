# export JT_SYNC=1
# export trace_py_var=3
 
python wiki2bio_jittor/src/main.py --mode train --batch_size 2 --batch_size_valid 2 --accumulation 16 --use_cuda False
