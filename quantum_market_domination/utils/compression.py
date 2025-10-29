"""
DATA COMPRESSION UTILITIES
Advanced compression techniques for efficient data storage and transmission
"""

import zlib
import gzip
import bz2
import lzma
import json
import pickle
from typing import Any, Dict, Union
import logging


class CompressionMethod:
    """Available compression methods"""
    ZLIB = "zlib"
    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"


class DataCompressor:
    """
    Data compression utilities supporting multiple algorithms
    """
    
    def __init__(self, default_method: str = CompressionMethod.ZLIB, 
                 compression_level: int = 9):
        """
        Initialize data compressor
        
        Args:
            default_method: Default compression method
            compression_level: Compression level (1-9, higher = better compression)
        """
        self.logger = logging.getLogger('DataCompressor')
        self.default_method = default_method
        self.compression_level = min(9, max(1, compression_level))
        
    def compress_string(self, data: str, method: str = None) -> bytes:
        """
        Compress string data
        
        Args:
            data: String to compress
            method: Compression method (defaults to default_method)
            
        Returns:
            Compressed bytes
        """
        if method is None:
            method = self.default_method
            
        data_bytes = data.encode('utf-8')
        
        if method == CompressionMethod.ZLIB:
            return zlib.compress(data_bytes, level=self.compression_level)
        elif method == CompressionMethod.GZIP:
            return gzip.compress(data_bytes, compresslevel=self.compression_level)
        elif method == CompressionMethod.BZ2:
            return bz2.compress(data_bytes, compresslevel=self.compression_level)
        elif method == CompressionMethod.LZMA:
            return lzma.compress(data_bytes, preset=self.compression_level)
        else:
            self.logger.warning(f"Unknown compression method: {method}, using zlib")
            return zlib.compress(data_bytes, level=self.compression_level)
    
    def decompress_string(self, compressed_data: bytes, method: str = None) -> str:
        """
        Decompress string data
        
        Args:
            compressed_data: Compressed bytes
            method: Compression method (defaults to default_method)
            
        Returns:
            Decompressed string
        """
        if method is None:
            method = self.default_method
            
        if method == CompressionMethod.ZLIB:
            data_bytes = zlib.decompress(compressed_data)
        elif method == CompressionMethod.GZIP:
            data_bytes = gzip.decompress(compressed_data)
        elif method == CompressionMethod.BZ2:
            data_bytes = bz2.decompress(compressed_data)
        elif method == CompressionMethod.LZMA:
            data_bytes = lzma.decompress(compressed_data)
        else:
            self.logger.warning(f"Unknown compression method: {method}, using zlib")
            data_bytes = zlib.decompress(compressed_data)
            
        return data_bytes.decode('utf-8')
    
    def compress_dict(self, data: Dict, method: str = None) -> bytes:
        """
        Compress dictionary data (via JSON)
        
        Args:
            data: Dict to compress
            method: Compression method
            
        Returns:
            Compressed bytes
        """
        json_str = json.dumps(data)
        return self.compress_string(json_str, method)
    
    def decompress_dict(self, compressed_data: bytes, method: str = None) -> Dict:
        """
        Decompress dictionary data
        
        Args:
            compressed_data: Compressed bytes
            method: Compression method
            
        Returns:
            Decompressed dict
        """
        json_str = self.decompress_string(compressed_data, method)
        return json.loads(json_str)
    
    def compress_object(self, obj: Any, method: str = None) -> bytes:
        """
        Compress Python object (via pickle)
        
        Args:
            obj: Object to compress
            method: Compression method
            
        Returns:
            Compressed bytes
        """
        pickled = pickle.dumps(obj)
        
        if method is None:
            method = self.default_method
            
        if method == CompressionMethod.ZLIB:
            return zlib.compress(pickled, level=self.compression_level)
        elif method == CompressionMethod.GZIP:
            return gzip.compress(pickled, compresslevel=self.compression_level)
        elif method == CompressionMethod.BZ2:
            return bz2.compress(pickled, compresslevel=self.compression_level)
        elif method == CompressionMethod.LZMA:
            return lzma.compress(pickled, preset=self.compression_level)
        else:
            return zlib.compress(pickled, level=self.compression_level)
    
    def decompress_object(self, compressed_data: bytes, method: str = None) -> Any:
        """
        Decompress Python object
        
        Args:
            compressed_data: Compressed bytes
            method: Compression method
            
        Returns:
            Decompressed object
        """
        if method is None:
            method = self.default_method
            
        if method == CompressionMethod.ZLIB:
            pickled = zlib.decompress(compressed_data)
        elif method == CompressionMethod.GZIP:
            pickled = gzip.decompress(compressed_data)
        elif method == CompressionMethod.BZ2:
            pickled = bz2.decompress(compressed_data)
        elif method == CompressionMethod.LZMA:
            pickled = lzma.decompress(compressed_data)
        else:
            pickled = zlib.decompress(compressed_data)
            
        return pickle.loads(pickled)
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        Calculate compression ratio
        
        Args:
            original_size: Original data size in bytes
            compressed_size: Compressed data size in bytes
            
        Returns:
            Compression ratio (higher = better compression)
        """
        if compressed_size == 0:
            return 0.0
        return original_size / compressed_size
    
    def compare_methods(self, data: str) -> Dict[str, Dict]:
        """
        Compare compression methods on given data
        
        Args:
            data: Data to compress
            
        Returns:
            Dict with compression statistics for each method
        """
        original_size = len(data.encode('utf-8'))
        results = {}
        
        methods = [
            CompressionMethod.ZLIB,
            CompressionMethod.GZIP,
            CompressionMethod.BZ2,
            CompressionMethod.LZMA
        ]
        
        for method in methods:
            try:
                import time
                start_time = time.time()
                compressed = self.compress_string(data, method)
                compress_time = time.time() - start_time
                
                compressed_size = len(compressed)
                ratio = self.get_compression_ratio(original_size, compressed_size)
                
                start_time = time.time()
                self.decompress_string(compressed, method)
                decompress_time = time.time() - start_time
                
                results[method] = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': ratio,
                    'space_saved_pct': (1 - compressed_size / original_size) * 100,
                    'compress_time': compress_time,
                    'decompress_time': decompress_time
                }
            except Exception as e:
                self.logger.error(f"Error testing {method}: {e}")
                results[method] = {'error': str(e)}
        
        return results
    
    def compress_market_data(self, market_data: Dict) -> bytes:
        """
        Compress market data with optimized settings
        
        Args:
            market_data: Market data dict
            
        Returns:
            Compressed bytes
        """
        # Use LZMA for best compression of structured market data
        return self.compress_dict(market_data, method=CompressionMethod.LZMA)
    
    def decompress_market_data(self, compressed_data: bytes) -> Dict:
        """
        Decompress market data
        
        Args:
            compressed_data: Compressed bytes
            
        Returns:
            Market data dict
        """
        return self.decompress_dict(compressed_data, method=CompressionMethod.LZMA)
    
    def compress_time_series(self, time_series_data: list) -> bytes:
        """
        Compress time series data with optimized settings
        
        Args:
            time_series_data: List of time series data points
            
        Returns:
            Compressed bytes
        """
        # Convert to JSON and compress with ZLIB (fast and good for numeric data)
        json_str = json.dumps(time_series_data)
        return self.compress_string(json_str, method=CompressionMethod.ZLIB)
    
    def decompress_time_series(self, compressed_data: bytes) -> list:
        """
        Decompress time series data
        
        Args:
            compressed_data: Compressed bytes
            
        Returns:
            Time series data list
        """
        json_str = self.decompress_string(compressed_data, method=CompressionMethod.ZLIB)
        return json.loads(json_str)
