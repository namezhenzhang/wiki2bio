import logging
import sys
APP_LOGGER_NAME = 'wiki2bio'
def setup_applevel_logger(logger_name = APP_LOGGER_NAME, file_name=None): 
    '''
    1.创建logger
    2.创建handler
    3.定义formatter
    4.给handler添加formatter
    5.给logger添加handler
    '''
    #1.创建一个logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    #3.定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    #2.创建一个handler，用于输出到控制台
    sh = logging.StreamHandler(sys.stdout)
    #4.给handler添加formatter
    sh.setFormatter(formatter)
    logger.handlers.clear()
    #5.给logger添加handler 
    logger.addHandler(sh)
    if file_name:
        #2.创建一个handler，用于写入日志文件
        fh = logging.FileHandler(file_name)
        #4.给handler添加formatter
        fh.setFormatter(formatter)
        #5.给logger添加handler 
        logger.addHandler(fh)
    return logger

def get_logger(module_name):    
    '''
    类似于prompt.configs这样的name
    '''
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)