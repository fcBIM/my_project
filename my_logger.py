from loguru import logger
logger.add("my_log.log", level="DEBUG", rotation="100 MB")
