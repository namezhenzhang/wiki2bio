import argparse
import os
import sys
import logger

log = logger.get_logger(__name__)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
_GLOBAL_ARGS = None
def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass  #error condition maybe?
def get_args_parser():
    
    parser = argparse.ArgumentParser(description="Command line interface for wiki2bio.")

    #==============Required parameters===========#
    parser.add_argument("--output_dir_base", default="outputlogs/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument('--seed', type=int, default=100,help="random seed for reproducing")
    parser.add_argument("--hidden_size",type=int,default=500,help="Size of each layer.")
    parser.add_argument("--emb_size",type=int,default=400,help="Size of embedding.")
    parser.add_argument("--field_size",type=int,default=50,help="Size of embedding.")
    parser.add_argument("--pos_size",type=int,default=5,help="Size of embedding.")
    parser.add_argument("--batch_size",type=int,default=32,help="Batch size of train set.")
    parser.add_argument("--batch_size_valid",type=int,default=128,help="Batch size of valid set.")
    parser.add_argument("--accumulation",type=int,default=1,help=".")
    parser.add_argument("--epoch",type=int,default=50,help="Number of training epoch.")
    parser.add_argument("--source_vocab",type=int,default=20003,help='vocabulary size')
    parser.add_argument("--field_vocab",type=int,default=1480,help='vocabulary size')
    parser.add_argument("--position_vocab",type=int,default=31,help='vocabulary size')
    parser.add_argument("--target_vocab",type=int,default=20003,help='vocabulary size')
    parser.add_argument("--report",type=int,default=5000,help='report valid results after some steps')
    parser.add_argument("--learning_rate",type=float,default=0.0003,help='learning rate')

    parser.add_argument("--mode",type=str,default='train',choices=['train','test'],help='train or test')
    parser.add_argument('--resume', type=str, default=None,help=" ")
    parser.add_argument("--load",type=str,default='0',help='load directory')
    parser.add_argument("--dir",type=str,default='processed_data',help='data set directory')
    parser.add_argument("--limits",type=int,default=0,help='max data set size')
    parser.add_argument("--use_cuda",type=t_or_f,default=False,help='')
    
    parser.add_argument("--dual_attention",type=t_or_f,default=True,help='dual attention layer or normal attention')
    parser.add_argument("--fgate_encoder",type=t_or_f,default=True,help='add field gate in encoder lstm')

    parser.add_argument("--field",type=t_or_f,default=False,help='concat field information to word embedding')
    parser.add_argument("--position",type=t_or_f,default=False,help='concat position information to word embedding')
    parser.add_argument("--encoder_pos",type=t_or_f,default=True,help='position information in field-gated encoder')
    parser.add_argument("--decoder_pos",type=t_or_f,default=True,help='position information in dual attention decoder')

    # tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
    # tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
    # tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
    # tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
    # tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
    # tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")
    # tf.app.flags.DEFINE_integer("source_vocab", 20003,'vocabulary size')
    # tf.app.flags.DEFINE_integer("field_vocab", 1480,'vocabulary size')
    # tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
    # tf.app.flags.DEFINE_integer("target_vocab", 20003,'vocabulary size')
    # tf.app.flags.DEFINE_integer("report", 5000,'report valid results after some steps')
    # tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

    # tf.app.flags.DEFINE_string("mode",'train','train or test')
    # tf.app.flags.DEFINE_string("load",'0','load directory') # BBBBBESTOFAll
    # tf.app.flags.DEFINE_string("dir",'processed_data','data set directory')
    # tf.app.flags.DEFINE_integer("limits", 0,'max data set size')

    # tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')
    # tf.app.flags.DEFINE_boolean("fgate_encoder", True,'add field gate in encoder lstm')

    # tf.app.flags.DEFINE_boolean("field", False,'concat field information to word embedding')
    # tf.app.flags.DEFINE_boolean("position", False,'concat position information to word embedding')
    # tf.app.flags.DEFINE_boolean("encoder_pos", True,'position information in field-gated encoder')
    # tf.app.flags.DEFINE_boolean("decoder_pos", True,'position information in dual attention decoder')


    #============================================#
    args = parser.parse_args()
    args.root_dir = root_dir
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
    random_code = get_next_global_code(args.output_dir_base)
    args.random_code = str(random_code)
    args.output_dir = os.path.join(args.output_dir_base, random_code)
    log.info(f"output dir {args.output_dir}")
    os.mkdir(args.output_dir)
    args.logger_file_name = os.path.join(args.output_dir,"output.log")