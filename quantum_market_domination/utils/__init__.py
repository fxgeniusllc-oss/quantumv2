"""Utilities module initialization"""

from .encryption import EncryptionUtils
from .compression import CompressionEngine
from .distributed_lock import DistributedLock, LockManager, get_lock_manager

# Alias for backward compatibility
AdvancedEncryption = EncryptionUtils

__all__ = [
    'EncryptionUtils',
    'AdvancedEncryption',
    'CompressionEngine',
    'DistributedLock',
    'LockManager',
    'get_lock_manager'
]
