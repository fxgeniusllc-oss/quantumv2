"""
ENCRYPTION UTILITIES
Advanced encryption for sensitive data
"""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import os


class EncryptionUtils:
    """
    Advanced encryption utilities
    """
    
    @staticmethod
    def generate_key():
        """Generate a new encryption key"""
        return Fernet.generate_key()

    @staticmethod
    def derive_key_from_password(password: str, salt: bytes = None) -> tuple:
        """
        Derive encryption key from password
        
        Args:
            password: Password string
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt

    @staticmethod
    def encrypt(data: bytes, key: bytes) -> bytes:
        """Encrypt data"""
        cipher_suite = Fernet(key)
        return cipher_suite.encrypt(data)

    @staticmethod
    def decrypt(encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data"""
        cipher_suite = Fernet(key)
        return cipher_suite.decrypt(encrypted_data)

    @staticmethod
    def encrypt_string(text: str, key: bytes) -> str:
        """Encrypt string and return base64 encoded"""
        encrypted = EncryptionUtils.encrypt(text.encode(), key)
        return base64.urlsafe_b64encode(encrypted).decode()

    @staticmethod
    def decrypt_string(encrypted_text: str, key: bytes) -> str:
        """Decrypt base64 encoded string"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode())
        decrypted = EncryptionUtils.decrypt(encrypted_bytes, key)
        return decrypted.decode()
