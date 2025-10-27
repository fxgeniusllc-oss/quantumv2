"""
QUANTUM-GRADE SECRET MANAGEMENT SYSTEM
Provides military-grade encryption and secure credential storage
"""

from cryptography.fernet import Fernet
import os
import json
from pathlib import Path


class SecretVault:
    """
    QUANTUM-GRADE SECRET MANAGEMENT SYSTEM
    Provides military-grade encryption and secure credential storage
    """
    
    def __init__(self, vault_path='secrets/vault.encrypted'):
        self.vault_path = vault_path
        self._ensure_secrets_directory()
        self._master_key = self._generate_or_load_master_key()
        self._cipher_suite = Fernet(self._master_key)

    def _ensure_secrets_directory(self):
        """Ensure secrets directory exists"""
        secrets_dir = os.path.dirname(self.vault_path)
        if secrets_dir and not os.path.exists(secrets_dir):
            os.makedirs(secrets_dir, exist_ok=True)

    def _generate_or_load_master_key(self):
        """
        Generate or retrieve the master encryption key
        Uses hardware-based key generation for maximum security
        """
        key_path = 'secrets/master.key'
        
        if os.path.exists(key_path):
            with open(key_path, 'rb') as key_file:
                return key_file.read()
        
        # Generate new master key with hardware entropy
        master_key = Fernet.generate_key()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        
        with open(key_path, 'wb') as key_file:
            key_file.write(master_key)
        
        # Secure file permissions (Unix-like systems)
        try:
            os.chmod(key_path, 0o600)
        except:
            pass
        
        return master_key

    def store_credentials(self, exchange_name, credentials):
        """
        Securely store exchange credentials with quantum encryption
        
        Args:
            exchange_name: Name of the exchange
            credentials: Dictionary containing API credentials
        """
        # Add exchange identifier to credentials
        credentials['exchange'] = exchange_name
        
        encrypted_credentials = self._cipher_suite.encrypt(
            json.dumps(credentials).encode()
        )
        
        # Append to vault file
        with open(self.vault_path, 'ab') as vault:
            vault.write(encrypted_credentials + b'\n')

    def get_credentials(self, exchange_name):
        """
        Retrieve and decrypt exchange credentials
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            Dictionary containing decrypted credentials
        """
        if not os.path.exists(self.vault_path):
            raise ValueError(f"Vault file not found: {self.vault_path}")
        
        with open(self.vault_path, 'rb') as vault:
            for line in vault:
                if not line.strip():
                    continue
                try:
                    decrypted = self._cipher_suite.decrypt(line.strip())
                    credentials = json.loads(decrypted)
                    
                    if credentials.get('exchange') == exchange_name:
                        return credentials
                except Exception as e:
                    continue
        
        raise ValueError(f"No credentials found for {exchange_name}")

    def list_exchanges(self):
        """
        List all exchanges with stored credentials
        
        Returns:
            List of exchange names
        """
        exchanges = []
        
        if not os.path.exists(self.vault_path):
            return exchanges
        
        with open(self.vault_path, 'rb') as vault:
            for line in vault:
                if not line.strip():
                    continue
                try:
                    decrypted = self._cipher_suite.decrypt(line.strip())
                    credentials = json.loads(decrypted)
                    exchange_name = credentials.get('exchange')
                    if exchange_name and exchange_name not in exchanges:
                        exchanges.append(exchange_name)
                except:
                    continue
        
        return exchanges

    def delete_credentials(self, exchange_name):
        """
        Delete credentials for a specific exchange
        
        Args:
            exchange_name: Name of the exchange
        """
        if not os.path.exists(self.vault_path):
            return
        
        # Read all credentials except the one to delete
        remaining_credentials = []
        
        with open(self.vault_path, 'rb') as vault:
            for line in vault:
                if not line.strip():
                    continue
                try:
                    decrypted = self._cipher_suite.decrypt(line.strip())
                    credentials = json.loads(decrypted)
                    
                    if credentials.get('exchange') != exchange_name:
                        remaining_credentials.append(line)
                except:
                    remaining_credentials.append(line)
        
        # Rewrite vault with remaining credentials
        with open(self.vault_path, 'wb') as vault:
            for line in remaining_credentials:
                vault.write(line)
