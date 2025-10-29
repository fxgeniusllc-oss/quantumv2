"""
WebSocket Manager
Handles WebSocket connections and streaming for multiple exchanges
"""

import asyncio
import websockets
import json
import zlib
import logging
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass


@dataclass
class WebSocketConfig:
    """WebSocket connection configuration"""
    url: str
    timeout: float = 30.0
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    compression_enabled: bool = True


class WebSocketManager:
    """
    Advanced WebSocket connection manager
    Handles multiple concurrent connections with automatic reconnection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger('WebSocketManager')
        self.config = config or {}
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.handlers: Dict[str, Callable] = {}
        self.active = False
        self.reconnect_attempts: Dict[str, int] = {}
        
        # Default configuration
        self.default_timeout = self.config.get('WEBSOCKET_TIMEOUT', 500) / 1000.0  # Convert ms to seconds
        self.max_reconnect_attempts = self.config.get('MAX_RECONNECT_ATTEMPTS', 5)
        self.compression_level = self.config.get('COMPRESSION_LEVEL', 9)
        
    async def connect(self, connection_id: str, ws_config: WebSocketConfig, handler: Callable):
        """
        Establish WebSocket connection with handler
        
        Args:
            connection_id: Unique identifier for this connection
            ws_config: WebSocket configuration
            handler: Async callback function for message handling
        """
        self.handlers[connection_id] = handler
        self.reconnect_attempts[connection_id] = 0
        
        await self._establish_connection(connection_id, ws_config)
        
    async def _establish_connection(self, connection_id: str, ws_config: WebSocketConfig):
        """
        Internal method to establish and maintain connection
        """
        while self.reconnect_attempts[connection_id] < self.max_reconnect_attempts:
            try:
                # Set up compression if enabled
                compression = 'deflate' if ws_config.compression_enabled else None
                
                async with websockets.connect(
                    ws_config.url,
                    ping_interval=20,
                    ping_timeout=ws_config.timeout,
                    compression=compression
                ) as websocket:
                    self.connections[connection_id] = websocket
                    self.reconnect_attempts[connection_id] = 0
                    self.logger.info(f"WebSocket connected: {connection_id}")
                    
                    # Message receiving loop
                    await self._message_loop(connection_id, websocket)
                    
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.WebSocketException) as e:
                self.logger.warning(f"WebSocket {connection_id} disconnected: {e}")
                self.reconnect_attempts[connection_id] += 1
                
                if self.reconnect_attempts[connection_id] < self.max_reconnect_attempts:
                    await asyncio.sleep(ws_config.reconnect_delay * self.reconnect_attempts[connection_id])
                    self.logger.info(f"Reconnecting {connection_id} (attempt {self.reconnect_attempts[connection_id]})")
                else:
                    self.logger.error(f"Max reconnection attempts reached for {connection_id}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Unexpected error in WebSocket {connection_id}: {e}")
                await asyncio.sleep(ws_config.reconnect_delay)
                
    async def _message_loop(self, connection_id: str, websocket):
        """
        Main message receiving loop
        """
        handler = self.handlers.get(connection_id)
        
        while self.active or not self.active:  # Run until connection closes
            try:
                raw_message = await websocket.recv()
                
                # Process message
                if handler:
                    await handler(connection_id, raw_message)
                    
            except websockets.exceptions.ConnectionClosed:
                self.logger.info(f"Connection {connection_id} closed")
                break
            except Exception as e:
                self.logger.error(f"Error processing message for {connection_id}: {e}")
                
    async def send(self, connection_id: str, message: dict):
        """
        Send message through WebSocket connection
        
        Args:
            connection_id: Target connection identifier
            message: Message dictionary to send
        """
        websocket = self.connections.get(connection_id)
        if websocket:
            try:
                await websocket.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error sending message to {connection_id}: {e}")
        else:
            self.logger.warning(f"No active connection for {connection_id}")
            
    async def disconnect(self, connection_id: str):
        """
        Close specific WebSocket connection
        """
        websocket = self.connections.get(connection_id)
        if websocket:
            await websocket.close()
            del self.connections[connection_id]
            self.logger.info(f"Disconnected {connection_id}")
            
    async def disconnect_all(self):
        """
        Close all active WebSocket connections
        """
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id)
            
    def decompress_message(self, compressed_data: bytes) -> dict:
        """
        Decompress and parse WebSocket message
        
        Args:
            compressed_data: Compressed message data
            
        Returns:
            Parsed message dictionary
        """
        try:
            decompressed = zlib.decompress(compressed_data)
            return json.loads(decompressed)
        except Exception as e:
            self.logger.error(f"Error decompressing message: {e}")
            return {}
            
    def is_connected(self, connection_id: str) -> bool:
        """
        Check if connection is active
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            True if connection is active
        """
        websocket = self.connections.get(connection_id)
        return websocket is not None and websocket.open
