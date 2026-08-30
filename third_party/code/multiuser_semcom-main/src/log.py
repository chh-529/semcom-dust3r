import logging
import sys
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that uses tqdm.write() to prevent log messages
    from interfering with tqdm progress bars.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def get_logger(logger_name: str = None, log_file_path: str | None = None, stdout: bool = False, stdout_tqdm_write: bool = True):
    """
        Args:
            logger_name: the name of logger
            log_file_path: the file path the logger writes to, with mode 'a'
                           can be None (don't log to file)
            stdout: whether to tee the log to standout.
            stdout_tqdm_write: whether to use tqdm.write for stdout
        
            
    """
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(filename)s[line:%(lineno)d]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if log_file_path is not None:
        handler = logging.FileHandler(log_file_path)
        handler.setFormatter(log_formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    
    if stdout:
        if stdout_tqdm_write:
            stdout_handler = TqdmLoggingHandler()
        else:
            stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_formatter)
        stdout_handler.setLevel(logging.DEBUG)
        logger.addHandler(stdout_handler)
    
    logger.info('logger init finished')
    return logger

if __name__ == '__main__':
    from . import utils
    logger = get_logger('main_logger', None, False, False)
    logger.info('abc')
    # logger.info(utils.toColor('avcdfiouashjoi', 'blue'))
    
    # import time
    # progress_bar = tqdm(range(10))
    # for i in progress_bar:
    #     for i in tqdm(range(10), leave=False):
    #         logger.info(i)
    #         progress_bar.set_description(f'msg {i}')
    #         time.sleep(0.5)