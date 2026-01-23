"""
Logging utilities for NeuroFormer.

Provides consistent logging across all modules.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


# Default format
DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level logger cache
_loggers = {}


def get_logger(
    name: str = "neuroformer",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (defaults to 'neuroformer')
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    _loggers[name] = logger
    return logger


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for the entire package.
    
    Args:
        level: Logging level
        log_file: Optional path to log file
        log_dir: Optional directory for log files (auto-generates filename)
        
    Returns:
        Root logger for neuroformer package
    """
    logger = get_logger("neuroformer", level)
    
    # Add file handler if specified
    if log_file or log_dir:
        if log_dir and not log_file:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(Path(log_dir) / f"neuroformer_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


class LoggerMixin:
    """
    Mixin class that provides logging capabilities.
    
    Classes inheriting from this mixin get a `logger` property.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(
                f"neuroformer.{self.__class__.__name__}"
            )
        return self._logger


def log_exception(logger: logging.Logger, exc: Exception, context: str = ""):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exc: Exception to log
        context: Optional context description
    """
    import traceback
    
    msg = f"Exception in {context}: {exc}" if context else str(exc)
    logger.error(msg)
    logger.debug(traceback.format_exc())
