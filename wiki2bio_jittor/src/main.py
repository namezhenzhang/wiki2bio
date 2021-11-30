
import logger
from configs import get_args_parser,  get_args, save_args
import random
import numpy  as np
import sys
sys.path.append('./')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)

def do_before_running():
    args = get_args_parser()
    logger.setup_applevel_logger(file_name = args.logger_file_name)
    log = logger.get_logger(__name__)
    log.info(f"output dir: '{args.output_dir}'")
    save_args()
    
    
    log.info(f"set seed {args.seed}")
    set_seed(args.seed)
    return args,log

if __name__=='__main__':
    args, log = do_before_running()
