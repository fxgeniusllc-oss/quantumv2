"""
DISTRIBUTED LOCKING MECHANISM
Provides thread-safe and process-safe locking for distributed systems
"""

import threading
import time
import os
import fcntl
from typing import Optional
from contextlib import contextmanager
import logging


class LockTimeout(Exception):
    """Exception raised when lock acquisition times out"""
    pass


class DistributedLock:
    """
    Distributed locking mechanism using file-based locks
    Supports both thread-level and process-level locking
    """
    
    def __init__(self, lock_name: str, lock_dir: str = "/tmp/quantum_locks",
                 timeout: float = 30.0):
        """
        Initialize distributed lock
        
        Args:
            lock_name: Unique name for this lock
            lock_dir: Directory to store lock files
            timeout: Maximum time to wait for lock acquisition (seconds)
        """
        self.logger = logging.getLogger('DistributedLock')
        self.lock_name = lock_name
        self.lock_dir = lock_dir
        self.timeout = timeout
        
        # Create lock directory if it doesn't exist
        os.makedirs(lock_dir, exist_ok=True)
        
        # Lock file path
        self.lock_file_path = os.path.join(lock_dir, f"{lock_name}.lock")
        
        # Thread lock for thread-safety within same process
        self._thread_lock = threading.RLock()
        
        # File handle for file lock
        self._file_handle: Optional[int] = None
        
        # Track acquisition
        self._acquired = False
        self._acquire_count = 0
        
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock
        
        Args:
            blocking: Whether to block until lock is acquired
            timeout: Override default timeout (None = use default)
            
        Returns:
            True if lock acquired, False otherwise
            
        Raises:
            LockTimeout: If timeout expires while waiting for lock
        """
        if timeout is None:
            timeout = self.timeout
            
        # First acquire thread lock
        if blocking:
            thread_acquired = self._thread_lock.acquire(blocking=True, timeout=timeout)
        else:
            thread_acquired = self._thread_lock.acquire(blocking=False)
        
        if not thread_acquired:
            return False
        
        try:
            # Check if already acquired (reentrant)
            if self._acquired:
                self._acquire_count += 1
                return True
            
            # Try to acquire file lock
            start_time = time.time()
            
            while True:
                try:
                    # Open lock file
                    self._file_handle = os.open(
                        self.lock_file_path,
                        os.O_CREAT | os.O_WRONLY | os.O_TRUNC
                    )
                    
                    # Try to acquire exclusive lock
                    if blocking:
                        # Calculate remaining timeout
                        elapsed = time.time() - start_time
                        remaining = timeout - elapsed
                        
                        if remaining <= 0:
                            raise LockTimeout(f"Lock acquisition timed out after {timeout}s")
                        
                        # Set alarm for timeout (Unix-like systems)
                        try:
                            fcntl.flock(self._file_handle, fcntl.LOCK_EX)
                            break
                        except (BlockingIOError, IOError):
                            time.sleep(0.1)
                            if time.time() - start_time > timeout:
                                raise LockTimeout(f"Lock acquisition timed out after {timeout}s")
                    else:
                        # Non-blocking acquisition
                        fcntl.flock(self._file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                        
                except (BlockingIOError, IOError) as e:
                    if not blocking:
                        self._cleanup_file_handle()
                        self._thread_lock.release()
                        return False
                    
                    # Check timeout
                    if time.time() - start_time > timeout:
                        self._cleanup_file_handle()
                        self._thread_lock.release()
                        raise LockTimeout(f"Lock acquisition timed out after {timeout}s")
                    
                    time.sleep(0.1)
            
            # Write PID to lock file for debugging
            os.write(self._file_handle, f"{os.getpid()}\n".encode())
            
            self._acquired = True
            self._acquire_count = 1
            self.logger.debug(f"Lock '{self.lock_name}' acquired by PID {os.getpid()}")
            return True
            
        except Exception as e:
            self._cleanup_file_handle()
            self._thread_lock.release()
            raise
    
    def release(self):
        """
        Release the lock
        
        Raises:
            RuntimeError: If lock is not acquired
        """
        if not self._acquired:
            raise RuntimeError("Cannot release lock that is not acquired")
        
        # Decrement acquire count (for reentrant locks)
        self._acquire_count -= 1
        
        if self._acquire_count > 0:
            # Still held by this thread
            return
        
        try:
            # Release file lock
            if self._file_handle is not None:
                try:
                    fcntl.flock(self._file_handle, fcntl.LOCK_UN)
                except Exception as e:
                    self.logger.error(f"Error releasing file lock: {e}")
                
                self._cleanup_file_handle()
            
            self._acquired = False
            self.logger.debug(f"Lock '{self.lock_name}' released by PID {os.getpid()}")
            
        finally:
            # Always release thread lock
            self._thread_lock.release()
    
    def _cleanup_file_handle(self):
        """Clean up file handle"""
        if self._file_handle is not None:
            try:
                os.close(self._file_handle)
            except Exception as e:
                self.logger.error(f"Error closing file handle: {e}")
            finally:
                self._file_handle = None
    
    @contextmanager
    def lock(self, timeout: Optional[float] = None):
        """
        Context manager for lock acquisition
        
        Args:
            timeout: Override default timeout
            
        Example:
            with distributed_lock.lock():
                # Critical section
                pass
        """
        acquired = self.acquire(timeout=timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()
    
    def is_locked(self) -> bool:
        """
        Check if lock is currently held by any process
        
        Returns:
            True if locked, False otherwise
        """
        if self._acquired:
            return True
        
        # Try non-blocking acquire to check
        try:
            fd = os.open(self.lock_file_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
            except (BlockingIOError, IOError):
                return True
            finally:
                os.close(fd)
        except FileNotFoundError:
            return False
    
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
        if self._acquired:
            try:
                self.release()
            except Exception:
                pass


class LockManager:
    """
    Manages multiple distributed locks
    """
    
    def __init__(self, lock_dir: str = "/tmp/quantum_locks"):
        """
        Initialize lock manager
        
        Args:
            lock_dir: Directory for lock files
        """
        self.logger = logging.getLogger('LockManager')
        self.lock_dir = lock_dir
        self.locks: dict = {}
        
    def get_lock(self, lock_name: str, timeout: float = 30.0) -> DistributedLock:
        """
        Get or create a distributed lock
        
        Args:
            lock_name: Name of the lock
            timeout: Lock timeout
            
        Returns:
            DistributedLock instance
        """
        if lock_name not in self.locks:
            self.locks[lock_name] = DistributedLock(
                lock_name=lock_name,
                lock_dir=self.lock_dir,
                timeout=timeout
            )
        
        return self.locks[lock_name]
    
    @contextmanager
    def lock(self, lock_name: str, timeout: Optional[float] = None):
        """
        Context manager for acquiring a named lock
        
        Args:
            lock_name: Name of the lock
            timeout: Lock timeout
            
        Example:
            with lock_manager.lock('trade_execution'):
                # Critical section
                pass
        """
        lock = self.get_lock(lock_name, timeout or 30.0)
        with lock.lock(timeout=timeout):
            yield lock
    
    def cleanup_stale_locks(self, max_age_seconds: int = 3600):
        """
        Clean up stale lock files
        
        Args:
            max_age_seconds: Maximum age for lock files (seconds)
        """
        if not os.path.exists(self.lock_dir):
            return
        
        current_time = time.time()
        removed_count = 0
        
        for filename in os.listdir(self.lock_dir):
            if not filename.endswith('.lock'):
                continue
            
            file_path = os.path.join(self.lock_dir, filename)
            
            try:
                # Check file age
                file_age = current_time - os.path.getmtime(file_path)
                
                if file_age > max_age_seconds:
                    # Try to remove stale lock
                    os.remove(file_path)
                    removed_count += 1
                    self.logger.info(f"Removed stale lock file: {filename}")
                    
            except Exception as e:
                self.logger.error(f"Error checking/removing lock file {filename}: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} stale lock files")
    
    def get_all_locks_status(self) -> dict:
        """
        Get status of all locks
        
        Returns:
            Dict mapping lock names to their status
        """
        status = {}
        
        for lock_name, lock in self.locks.items():
            status[lock_name] = {
                'acquired': lock._acquired,
                'locked': lock.is_locked(),
                'acquire_count': lock._acquire_count
            }
        
        return status


# Global lock manager instance
_global_lock_manager = None


def get_lock_manager() -> LockManager:
    """
    Get the global lock manager instance.
    Creates one if it doesn't exist.
    
    Returns:
        LockManager: Global lock manager instance
    """
    global _global_lock_manager
    if _global_lock_manager is None:
        _global_lock_manager = LockManager()
    return _global_lock_manager
