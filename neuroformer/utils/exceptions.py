"""
Custom exceptions for NeuroFormer.

Provides specific error types for better error handling and debugging.
"""


class NeuroFormerError(Exception):
    """Base exception for all NeuroFormer errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class DataValidationError(NeuroFormerError):
    """Raised when input data fails validation checks."""
    
    def __init__(self, message: str, expected=None, got=None, field: str = None):
        details = {}
        if expected is not None:
            details['expected'] = expected
        if got is not None:
            details['got'] = got
        if field:
            details['field'] = field
        super().__init__(message, details)


class ModelConfigError(NeuroFormerError):
    """Raised when model configuration is invalid."""
    
    def __init__(self, message: str, param: str = None, value=None):
        details = {}
        if param:
            details['param'] = param
        if value is not None:
            details['value'] = value
        super().__init__(message, details)


class CheckpointError(NeuroFormerError):
    """Raised when checkpoint loading/saving fails."""
    
    def __init__(self, message: str, path: str = None):
        details = {'path': path} if path else {}
        super().__init__(message, details)


class PreprocessingError(NeuroFormerError):
    """Raised when preprocessing operations fail."""
    pass


class InferenceError(NeuroFormerError):
    """Raised when inference operations fail."""
    pass


class TrainingError(NeuroFormerError):
    """Raised when training operations fail."""
    pass
