"""
Compression Utilities
Data compression techniques for efficient storage and transmission
"""

import zlib
import gzip
import bz2
import lzma
import pickle
import json
from typing import Any, Dict, Optional
import logging


class CompressionEngine:
    """
    Advanced data compression engine
    Supports multiple compression algorithms
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('CompressionEngine')
        self.config = config or {}
        
        # Default compression level (1-9, 9 is highest)
        self.compression_level = self.config.get('COMPRESSION_LEVEL', 9)
        
        # Available algorithms
        self.algorithms = {
            'zlib': {
                'compress': self._zlib_compress,
                'decompress': self._zlib_decompress
            },
            'gzip': {
                'compress': self._gzip_compress,
                'decompress': self._gzip_decompress
            },
            'bz2': {
                'compress': self._bz2_compress,
                'decompress': self._bz2_decompress
            },
            'lzma': {
                'compress': self._lzma_compress,
                'decompress': self._lzma_decompress
            }
        }
        
    def compress(self, data: Any, algorithm: str = 'zlib', serialize: bool = True) -> bytes:
        """
        Compress data using specified algorithm
        
        Args:
            data: Data to compress
            algorithm: Compression algorithm ('zlib', 'gzip', 'bz2', 'lzma')
            serialize: Whether to serialize data first (for non-string data)
            
        Returns:
            Compressed data as bytes
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        # Serialize data if needed
        if serialize:
            if isinstance(data, (dict, list)):
                data = json.dumps(data).encode('utf-8')
            elif not isinstance(data, bytes):
                data = str(data).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')
            
        # Compress
        compress_func = self.algorithms[algorithm]['compress']
        compressed = compress_func(data)
        
        # Log compression ratio
        original_size = len(data)
        compressed_size = len(compressed)
        ratio = compressed_size / original_size if original_size > 0 else 0
        
        self.logger.debug(f"Compressed {original_size} bytes to {compressed_size} bytes "
                         f"({ratio*100:.1f}%) using {algorithm}")
        
        return compressed
        
    def decompress(self, compressed_data: bytes, algorithm: str = 'zlib', 
                   deserialize: bool = True, as_json: bool = False) -> Any:
        """
        Decompress data
        
        Args:
            compressed_data: Compressed data
            algorithm: Compression algorithm used
            deserialize: Whether to deserialize after decompression
            as_json: Whether to parse as JSON
            
        Returns:
            Decompressed data
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        # Decompress
        decompress_func = self.algorithms[algorithm]['decompress']
        data = decompress_func(compressed_data)
        
        # Deserialize if needed
        if deserialize:
            if as_json:
                data = json.loads(data.decode('utf-8'))
            else:
                data = data.decode('utf-8')
                
        return data
        
    def _zlib_compress(self, data: bytes) -> bytes:
        """Compress using zlib"""
        return zlib.compress(data, level=self.compression_level)
        
    def _zlib_decompress(self, data: bytes) -> bytes:
        """Decompress using zlib"""
        return zlib.decompress(data)
        
    def _gzip_compress(self, data: bytes) -> bytes:
        """Compress using gzip"""
        return gzip.compress(data, compresslevel=self.compression_level)
        
    def _gzip_decompress(self, data: bytes) -> bytes:
        """Decompress using gzip"""
        return gzip.decompress(data)
        
    def _bz2_compress(self, data: bytes) -> bytes:
        """Compress using bz2"""
        return bz2.compress(data, compresslevel=self.compression_level)
        
    def _bz2_decompress(self, data: bytes) -> bytes:
        """Decompress using bz2"""
        return bz2.decompress(data)
        
    def _lzma_compress(self, data: bytes) -> bytes:
        """Compress using lzma"""
        return lzma.compress(data, preset=self.compression_level)
        
    def _lzma_decompress(self, data: bytes) -> bytes:
        """Decompress using lzma"""
        return lzma.decompress(data)
        
    def benchmark_algorithms(self, data: Any) -> Dict:
        """
        Benchmark all compression algorithms
        
        Args:
            data: Data to test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        # Serialize data once
        if isinstance(data, (dict, list)):
            serialized = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            serialized = data.encode('utf-8')
        else:
            serialized = str(data).encode('utf-8')
            
        original_size = len(serialized)
        
        for algorithm in self.algorithms.keys():
            try:
                import time
                
                # Test compression
                start = time.time()
                compressed = self.compress(serialized, algorithm=algorithm, serialize=False)
                compress_time = time.time() - start
                
                # Test decompression
                start = time.time()
                decompressed = self.decompress(compressed, algorithm=algorithm, deserialize=False)
                decompress_time = time.time() - start
                
                # Calculate metrics
                compressed_size = len(compressed)
                ratio = compressed_size / original_size
                
                results[algorithm] = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': ratio,
                    'space_saving_pct': (1 - ratio) * 100,
                    'compress_time_ms': compress_time * 1000,
                    'decompress_time_ms': decompress_time * 1000,
                    'roundtrip_correct': decompressed == serialized
                }
                
            except Exception as e:
                results[algorithm] = {'error': str(e)}
                
        return results
        
    def compress_file(self, input_path: str, output_path: str, algorithm: str = 'gzip'):
        """
        Compress a file
        
        Args:
            input_path: Input file path
            output_path: Output file path
            algorithm: Compression algorithm
        """
        with open(input_path, 'rb') as f:
            data = f.read()
            
        compressed = self.compress(data, algorithm=algorithm, serialize=False)
        
        with open(output_path, 'wb') as f:
            f.write(compressed)
            
        self.logger.info(f"Compressed {input_path} to {output_path}")
        
    def decompress_file(self, input_path: str, output_path: str, algorithm: str = 'gzip'):
        """
        Decompress a file
        
        Args:
            input_path: Input file path
            output_path: Output file path
            algorithm: Compression algorithm
        """
        with open(input_path, 'rb') as f:
            compressed = f.read()
            
        data = self.decompress(compressed, algorithm=algorithm, deserialize=False)
        
        with open(output_path, 'wb') as f:
            f.write(data)
            
        self.logger.info(f"Decompressed {input_path} to {output_path}")
        
    def get_optimal_algorithm(self, data: Any, priority: str = 'ratio') -> str:
        """
        Find optimal compression algorithm for given data
        
        Args:
            data: Data to analyze
            priority: Optimization priority ('ratio', 'speed', 'balanced')
            
        Returns:
            Best algorithm name
        """
        results = self.benchmark_algorithms(data)
        
        if priority == 'ratio':
            # Best compression ratio
            best = min(results.items(), 
                      key=lambda x: x[1].get('compression_ratio', float('inf'))
                      if 'error' not in x[1] else float('inf'))
        elif priority == 'speed':
            # Fastest compression
            best = min(results.items(),
                      key=lambda x: x[1].get('compress_time_ms', float('inf'))
                      if 'error' not in x[1] else float('inf'))
        elif priority == 'balanced':
            # Balance between ratio and speed
            best = min(results.items(),
                      key=lambda x: (x[1].get('compression_ratio', float('inf')) + 
                                   x[1].get('compress_time_ms', float('inf')) / 1000)
                      if 'error' not in x[1] else float('inf'))
        else:
            raise ValueError(f"Unknown priority: {priority}")
            
        return best[0]
