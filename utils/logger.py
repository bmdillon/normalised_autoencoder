import logging
import os

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
