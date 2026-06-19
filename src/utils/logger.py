import os
import logging
from logging import Logger

logDir = os.path.join(os.getcwd(),"logs")
os.makedirs(logDir, exist_ok=True)

def configLogger(loggerName: str, fileName: str) -> Logger:

    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if the logger is imported multiple times
    if logger.hasHandlers():
        return logger

    logFormat = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )

    filePath = os.path.join(logDir, fileName)
    
    fileHandler = logging.FileHandler(filePath)
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormat)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(logFormat)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger
