"""
log_levels_demo.py  —  Task 3: demonstrate each log level.
Run twice to see the difference:
    LOG_LEVEL=DEBUG python log_levels_demo.py   -> all 5 lines appear
    LOG_LEVEL=INFO  python log_levels_demo.py   -> DEBUG line is filtered out
"""
from logger_config import get_logger

logger = get_logger()

logger.debug("detailed diagnostics for developers")
logger.info("normal event - service handled a request")
logger.warning("unexpected input but still handled")
logger.error("a request failed")
logger.critical("service cannot continue")
