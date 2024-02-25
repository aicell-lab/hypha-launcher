def get_logger():
    try:
        from loguru import logger
    except ImportError:
        import logging

        logger = logging.getLogger(__name__)
    return logger
