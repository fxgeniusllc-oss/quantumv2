"""
Distributed Lock Mechanism
Provides distributed locking for coordinating operations across multiple processes/instances
"""

import time
import uuid
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import threading


class DistributedLock:
    """
    Distributed locking mechanism
    Ensures mutual exclusion across distributed systems
    """
    
    def __init__(self, lock_name: str, timeout: float = 30.0, auto_renew: bool = True):
        """
        Initialize distributed lock
        
        Args:
            lock_name: Name of the lock
            timeout: Lock timeout in seconds
            auto_renew: Whether to auto-renew the lock
        """
        self.lock_name = lock_name
        self.timeout = timeout
        self.auto_renew = auto_renew
        self.lock_id = str(uuid.uuid4())
        self.acquired = False
        self.logger = logging.getLogger('DistributedLock')
        
        # Renewal thread
        self._renewal_thread = None
        self._stop_renewal = threading.Event()
        
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock
        
        Args:
            blocking: Whether to block waiting for the lock
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            True if lock acquired, False otherwise
        """
        start_time = time.time()
        
        while True:
            # Try to acquire lock
            if self._try_acquire():
                self.acquired = True
                self.logger.info(f"Acquired lock: {self.lock_name} (ID: {self.lock_id})")
                
                # Start auto-renewal if enabled
                if self.auto_renew:
                    self._start_renewal()
                    
                return True
                
            if not blocking:
                return False
                
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                self.logger.warning(f"Failed to acquire lock {self.lock_name} within timeout")
                return False
                
            # Wait before retry
            time.sleep(0.1)
            
    def _try_acquire(self) -> bool:
        """
        Attempt to acquire lock
        In a real implementation, this would interact with Redis, etcd, or similar
        This is a simplified in-memory version for demonstration
        """
        # This is a placeholder - in production, use Redis SETNX or similar
        return True
        
    def release(self):
        """Release the lock"""
        if not self.acquired:
            self.logger.warning(f"Attempting to release unacquired lock: {self.lock_name}")
            return
            
        # Stop auto-renewal
        if self.auto_renew:
            self._stop_renewal_thread()
            
        # Release lock
        self._do_release()
        self.acquired = False
        self.logger.info(f"Released lock: {self.lock_name} (ID: {self.lock_id})")
        
    def _do_release(self):
        """
        Perform actual lock release
        In production, this would interact with Redis DEL or similar
        """
        pass
        
    def _start_renewal(self):
        """Start automatic lock renewal thread"""
        self._stop_renewal.clear()
        self._renewal_thread = threading.Thread(target=self._renewal_loop, daemon=True)
        self._renewal_thread.start()
        
    def _renewal_loop(self):
        """Lock renewal loop"""
        renewal_interval = self.timeout / 2  # Renew at half the timeout
        
        while not self._stop_renewal.is_set():
            time.sleep(renewal_interval)
            
            if self.acquired and not self._stop_renewal.is_set():
                self._renew_lock()
                
    def _renew_lock(self):
        """
        Renew the lock
        In production, this would extend the TTL in Redis or similar
        """
        self.logger.debug(f"Renewed lock: {self.lock_name}")
        
    def _stop_renewal_thread(self):
        """Stop the renewal thread"""
        if self._renewal_thread and self._renewal_thread.is_alive():
            self._stop_renewal.set()
            self._renewal_thread.join(timeout=2.0)
            
    def __enter__(self):
        """Context manager entry"""
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
        return False
        
    def __del__(self):
        """Cleanup on deletion"""
        if self.acquired:
            self.release()


class LockManager:
    """
    Manager for multiple distributed locks
    Provides centralized lock management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('LockManager')
        self.config = config or {}
        
        # Lock storage (in production, this would be Redis/etcd)
        self._locks: Dict[str, Dict] = {}
        self._lock_mutex = threading.Lock()
        
        # Configuration
        self.default_timeout = self.config.get('lock_timeout', 30.0)
        self.max_locks = self.config.get('max_locks', 100)
        
    def get_lock(self, lock_name: str, timeout: Optional[float] = None, 
                 auto_renew: bool = True) -> DistributedLock:
        """
        Get a distributed lock
        
        Args:
            lock_name: Name of the lock
            timeout: Lock timeout (None = use default)
            auto_renew: Whether to auto-renew
            
        Returns:
            DistributedLock object
        """
        timeout = timeout or self.default_timeout
        return DistributedLock(lock_name, timeout=timeout, auto_renew=auto_renew)
        
    def is_locked(self, lock_name: str) -> bool:
        """
        Check if a lock is currently held
        
        Args:
            lock_name: Name of the lock
            
        Returns:
            True if locked, False otherwise
        """
        with self._lock_mutex:
            if lock_name in self._locks:
                lock_data = self._locks[lock_name]
                expiry = lock_data.get('expiry')
                
                if expiry and datetime.now() < expiry:
                    return True
                else:
                    # Lock expired, clean it up
                    del self._locks[lock_name]
                    
            return False
            
    def acquire_lock(self, lock_name: str, lock_id: str, 
                    timeout: float) -> bool:
        """
        Acquire a lock (internal method)
        
        Args:
            lock_name: Name of the lock
            lock_id: Unique lock identifier
            timeout: Lock timeout in seconds
            
        Returns:
            True if acquired, False otherwise
        """
        with self._lock_mutex:
            # Check if lock exists and is not expired
            if self.is_locked(lock_name):
                return False
                
            # Check max locks limit
            if len(self._locks) >= self.max_locks:
                self.logger.warning(f"Max locks limit ({self.max_locks}) reached")
                return False
                
            # Acquire lock
            self._locks[lock_name] = {
                'lock_id': lock_id,
                'acquired_at': datetime.now(),
                'expiry': datetime.now() + timedelta(seconds=timeout)
            }
            
            return True
            
    def release_lock(self, lock_name: str, lock_id: str) -> bool:
        """
        Release a lock (internal method)
        
        Args:
            lock_name: Name of the lock
            lock_id: Lock identifier
            
        Returns:
            True if released, False otherwise
        """
        with self._lock_mutex:
            if lock_name in self._locks:
                lock_data = self._locks[lock_name]
                
                # Verify lock ownership
                if lock_data['lock_id'] == lock_id:
                    del self._locks[lock_name]
                    return True
                else:
                    self.logger.warning(f"Lock {lock_name} owned by different ID")
                    return False
                    
            return False
            
    def renew_lock(self, lock_name: str, lock_id: str, timeout: float) -> bool:
        """
        Renew a lock (internal method)
        
        Args:
            lock_name: Name of the lock
            lock_id: Lock identifier
            timeout: New timeout in seconds
            
        Returns:
            True if renewed, False otherwise
        """
        with self._lock_mutex:
            if lock_name in self._locks:
                lock_data = self._locks[lock_name]
                
                # Verify lock ownership
                if lock_data['lock_id'] == lock_id:
                    lock_data['expiry'] = datetime.now() + timedelta(seconds=timeout)
                    return True
                    
            return False
            
    def get_lock_info(self, lock_name: str) -> Optional[Dict]:
        """
        Get information about a lock
        
        Args:
            lock_name: Name of the lock
            
        Returns:
            Lock information dictionary or None
        """
        with self._lock_mutex:
            if lock_name in self._locks:
                lock_data = self._locks[lock_name].copy()
                
                # Add age information
                age = (datetime.now() - lock_data['acquired_at']).total_seconds()
                remaining = (lock_data['expiry'] - datetime.now()).total_seconds()
                
                lock_data['age_seconds'] = age
                lock_data['remaining_seconds'] = max(0, remaining)
                
                return lock_data
                
            return None
            
    def cleanup_expired_locks(self) -> int:
        """
        Clean up expired locks
        
        Returns:
            Number of locks cleaned up
        """
        with self._lock_mutex:
            now = datetime.now()
            expired = [name for name, data in self._locks.items() 
                      if data['expiry'] < now]
            
            for name in expired:
                del self._locks[name]
                
            if expired:
                self.logger.info(f"Cleaned up {len(expired)} expired locks")
                
            return len(expired)
            
    def get_all_locks(self) -> Dict[str, Dict]:
        """
        Get information about all locks
        
        Returns:
            Dictionary of all locks and their info
        """
        with self._lock_mutex:
            return {name: self.get_lock_info(name) 
                   for name in self._locks.keys()}
            
    def force_release_all(self):
        """Force release all locks (use with caution)"""
        with self._lock_mutex:
            count = len(self._locks)
            self._locks.clear()
            self.logger.warning(f"Force released all {count} locks")


# Global lock manager instance
_global_lock_manager = None


def get_lock_manager() -> LockManager:
    """Get global lock manager instance"""
    global _global_lock_manager
    
    if _global_lock_manager is None:
        _global_lock_manager = LockManager()
        
    return _global_lock_manager
