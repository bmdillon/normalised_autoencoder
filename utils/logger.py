import os
import sys
import logging

def get_logger(name=__name__, logfile='log_default.txt', level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        if os.path.exists(logfile):
            raise FileExistsError(f"Log file '{logfile}' already exists. Quitting to avoid overwrite.")

        os.makedirs(os.path.dirname(logfile), exist_ok=True)

        # console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # file handler
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def setup_exception_logging(logger):
    def handle_exception(exc_type, exc_value, exc_traceback):
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception
