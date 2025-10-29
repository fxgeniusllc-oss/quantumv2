"""Utilities module initialization"""

from .encryption import EncryptionUtils
from .compression import DataCompressor, CompressionMethod
from .distributed_lock import DistributedLock, LockManager, get_lock_manager

# Aliases for backward compatibility
AdvancedEncryption = EncryptionUtils
CompressionEngine = DataCompressor  # Alias for backward compatibility

__all__ = [
    'EncryptionUtils',
    'AdvancedEncryption',
    'DataCompressor',
    'CompressionEngine',
    'CompressionMethod',
    'DistributedLock',
    'LockManager',
    'get_lock_manager'
]
