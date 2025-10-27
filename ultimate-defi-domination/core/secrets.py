"""
SECRET MANAGEMENT FOR DEFI
Secure private key and credential management
"""

from cryptography.fernet import Fernet
import os


class SecretManager:
    """
    Secure secret management for DeFi operations
    """
    
    def __init__(self):
        self.key = self._load_or_generate_key()
        self.cipher_suite = Fernet(self.key)

    def _load_or_generate_key(self):
        """Load existing key or generate new one"""
        key_path = 'secrets/defi_master.key'
        
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                return f.read()
        
        key = Fernet.generate_key()
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        
        with open(key_path, 'wb') as f:
            f.write(key)
        
        try:
            os.chmod(key_path, 0o600)
        except:
            pass
        
        return key

    def encrypt(self, message: str) -> bytes:
        """Encrypt a message"""
        return self.cipher_suite.encrypt(message.encode())

    def decrypt(self, encrypted_message: bytes) -> str:
        """Decrypt a message"""
        return self.cipher_suite.decrypt(encrypted_message).decode()

    def load_private_keys(self):
        """Securely load and decrypt private keys"""
        # In production, load from encrypted vault
        # For now, load from environment
        return os.getenv('PRIVATE_KEY', '')

    def store_private_key(self, key: str, identifier: str):
        """Store encrypted private key"""
        encrypted = self.encrypt(key)
        
        key_file = f'secrets/pk_{identifier}.enc'
        with open(key_file, 'wb') as f:
            f.write(encrypted)
        
        try:
            os.chmod(key_file, 0o600)
        except:
            pass
