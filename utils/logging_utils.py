import logging

loggers = {} 

def create_logger(
    log_level:str ='INFO', 
    log_name:str = 'logfile',
    export_log: bool = True,
    save_dir:str = ''):
    if log_name in loggers.keys():
        logger = loggers.get(log_name)
    else:
        # create logger
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        handler1 = logging.StreamHandler()
        handler1.setLevel(log_level)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        handler1.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(handler1)

        if export_log:
            pathname = log_name
            if len(save_dir)>0:
                pathname = f'{save_dir}/{pathname}'

            handler2 = logging.FileHandler(filename=f'{pathname}.log', mode='w')
            handler2.setLevel('DEBUG')
            handler2.setFormatter(formatter)
            logger.addHandler(handler2)
            
        loggers[log_name] = logger
    
    return logger
