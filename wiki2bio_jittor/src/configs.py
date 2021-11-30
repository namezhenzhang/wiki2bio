import argparse
import os
import sys
import logger

log = logger.get_logger(__name__)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
_GLOBAL_ARGS = None
def get_args_parser():
    
    parser = argparse.ArgumentParser(description="Command line interface for wiki2bio.")

    #==============Required parameters===========#
    parser.add_argument("--output_dir_base", default="outputlogs/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument('--seed', type=int, default=100,
                        help="random seed for reproducing")




    #============================================#
    args = parser.parse_args()
    args.output_dir_base = os.path.join(root_dir,args.output_dir_base)

    if not os.path.exists(args.output_dir_base):
        os.mkdir(args.output_dir_base)

    change_output_dir(args)

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

    return args

def get_args():
    '''
    在get_args_parser中定义的全局变量
    '''
    return _GLOBAL_ARGS 

def save_args():
    '''
    保存args到configs.txt
    '''
    args = _GLOBAL_ARGS
    dic = args.__dict__
    with open(os.path.join(args.output_dir,"configs.txt"), 'w') as f:
        for key in dic:
            f.write("{}\t{}\n".format(key,dic[key]))
    log.info("config saved!")


def get_next_global_code(output_dir):
    '''
    自动增加文件名序号，不用每次改变文件名
    '''
    
    filelist = os.listdir(output_dir)#列出该文件夹下所有的文件和文件夹
    filecode = []
    for file in filelist:
        try:
            x = int(file)
        except ValueError:
            continue
        filecode.append(x)
    if len(filecode)>0:
        cur_code = max(filecode)+1
    else:
        cur_code = 0
    cur_code_str = str(cur_code).rjust(5,'0')
    return cur_code_str

def change_output_dir(args):
    '''下一个输出目录名'''
    print('hello')
    random_code = get_next_global_code(args.output_dir_base)
    args.random_code = str(random_code)
    args.output_dir = os.path.join(args.output_dir_base, random_code)
    log.info(f"output dir {args.output_dir}")
    os.mkdir(args.output_dir)
    args.logger_file_name = os.path.join(args.output_dir,"output.log")