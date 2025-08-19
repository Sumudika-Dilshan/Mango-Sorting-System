#!/usr/bin/env python3
"""
PLC Communication Module for Mango Sorting System
Handles serial/Ethernet communication with Siemens S7-314C 2PN/DP PLC

Features:
- Serial communication support
- Ethernet/Modbus TCP support
- Connection management and error handling
- Command queue for reliable transmission

Author: [Your Name]
Date: [Date]
Version: 1.0
"""

import serial
import time
import logging
import threading
from queue import Queue, Empty
from config import *

logger = logging.getLogger(__name__)

class PLCCommunicator:
    def __init__(self, communication_type='serial'):
        """
        Initialize PLC communication
        
        Args:
            communication_type (str): 'serial' or 'ethernet'
        """
        self.communication_type = communication_type
        self.connection = None
        self.connected = False
        self.command_queue = Queue()
        self.worker_thread = None
        self.stop_worker = threading.Event()
        
    def connect(self):
        """Establish connection to PLC"""
        try:
            if self.communication_type == 'serial':
                self._connect_serial()
            elif self.communication_type == 'ethernet':
                self._connect_ethernet()
            else:
                raise ValueError(f"Unsupported communication type: {self.communication_type}")
                
            self.connected = True
            self._start_worker_thread()
            logger.info(f"PLC connection established ({self.communication_type})")
            
        except Exception as e:
            logger.error(f"Failed to connect to PLC: {e}")
            raise
            
    def _connect_serial(self):
        """Establish serial connection to PLC"""
        self.connection = serial.Serial(
            port=PLC_PORT,
            baudrate=PLC_BAUDRATE,
            timeout=PLC_TIMEOUT,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        
        # Verify connection with handshake
        self.connection.write(b'HELLO\n')
        time.sleep(0.1)
        response = self.connection.readline()
        
        if not response:
            logger.warning("No response from PLC during handshake")
            
    def _connect_ethernet(self):
        """Establish Ethernet connection to PLC (Modbus TCP)"""
        try:
            from pymodbus.client.sync import ModbusTcpClient
            
            self.connection = ModbusTcpClient(
                host=PLC_IP_ADDRESS,
                port=PLC_PORT_NUMBER,
                timeout=CONNECTION_TIMEOUT
            )
            
            if not self.connection.connect():
                raise Exception("Failed to establish Modbus TCP connection")
                
        except ImportError:
            logger.error("pymodbus library not installed. Install with: pip install pymodbus")
            raise
        except Exception as e:
            logger.error(f"Ethernet connection failed: {e}")
            raise
            
    def _start_worker_thread(self):
        """Start background thread for command processing"""
        self.worker_thread = threading.Thread(target=self._command_worker, daemon=True)
        self.worker_thread.start()
        
    def _command_worker(self):
        """Background worker thread for processing command queue"""
        while not self.stop_worker.is_set():
            try:
                # Wait for command with timeout
                command = self.command_queue.get(timeout=1.0)
                
                # Send command to PLC
                self._send_command_direct(command)
                
                # Mark task as done
                self.command_queue.task_done()
                
            except Empty:
                continue  # Timeout, check if we should stop
            except Exception as e:
                logger.error(f"Command worker error: {e}")
                
    def send_command(self, command):
        """
        Queue command for transmission to PLC
        
        Args:
            command (bytes): Command to send to PLC
        """
        if not self.connected:
            logger.error("PLC not connected - cannot send command")
            return False
            
        try:
            self.command_queue.put(command, timeout=1.0)
            logger.debug(f"Command queued: {command}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue command: {e}")
            return False
            
    def _send_command_direct(self, command):
        """
        Send command directly to PLC
        
        Args:
            command (bytes): Command to send
        """
        try:
            if self.communication_type == 'serial':
                self._send_serial_command(command)
            elif self.communication_type == 'ethernet':
                self._send_ethernet_command(command)
                
            logger.info(f"Command sent to PLC: {command}")
            
        except Exception as e:
            logger.error(f"Failed to send command {command}: {e}")
            raise
            
    def _send_serial_command(self, command):
        """Send command via serial connection"""
        if not self.connection or not self.connection.is_open:
            raise Exception("Serial connection not available")
            
        self.connection.write(command)
        self.connection.flush()
        
        # Wait for acknowledgment (optional)
        time.sleep(0.05)
        if self.connection.in_waiting > 0:
            response = self.connection.readline()
            logger.debug(f"PLC response: {response}")
            
    def _send_ethernet_command(self, command):
        """Send command via Ethernet/Modbus TCP"""
        if not self.connection:
            raise Exception("Ethernet connection not available")
            
        # Convert command to Modbus register writes
        command_map = {
            COMMAND_RIPE: 1,
            COMMAND_SEMIRIPE: 2,
            COMMAND_UNRIPE: 3
        }
        
        register_value = command_map.get(command, 0)
        
        if register_value > 0:
            # Write to holding register (address 0 in this example)
            result = self.connection.write_register(0, register_value)
            
            if result.isError():
                raise Exception(f"Modbus write error: {result}")
                
    def send_heartbeat(self):
        """Send heartbeat signal to maintain PLC connection"""
        try:
            if self.communication_type == 'serial':
                self.send_command(b'HEARTBEAT\n')
            elif self.communication_type == 'ethernet':
                # Read a register to test connection
                result = self.connection.read_holding_registers(0, 1)
                if result.isError():
                    logger.warning("Heartbeat failed - connection may be lost")
                    
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            
    def get_status(self):
        """Get current status from PLC"""
        try:
            if self.communication_type == 'serial':
                self.connection.write(b'STATUS\n')
                time.sleep(0.1)
                response = self.connection.readline()
                return response.decode().strip()
                
            elif self.communication_type == 'ethernet':
                # Read status registers
                result = self.connection.read_holding_registers(10, 5)
                if not result.isError():
                    return result.registers
                    
        except Exception as e:
            logger.error(f"Failed to get PLC status: {e}")
            
        return None
        
    def reset_system(self):
        """Send reset command to PLC"""
        try:
            if self.communication_type == 'serial':
                self.send_command(b'RESET\n')
            elif self.communication_type == 'ethernet':
                self.connection.write_register(99, 1)  # Reset register
                
            logger.info("System reset command sent")
            
        except Exception as e:
            logger.error(f"Failed to send reset command: {e}")
            
    def emergency_stop(self):
        """Send emergency stop command"""
        try:
            if self.communication_type == 'serial':
                self.connection.write(b'EMERGENCY_STOP\n')
                self.connection.flush()
            elif self.communication_type == 'ethernet':
                self.connection.write_register(98, 1)  # Emergency stop register
                
            logger.critical("Emergency stop activated")
            
        except Exception as e:
            logger.error(f"Failed to send emergency stop: {e}")
            
    def disconnect(self):
        """Close PLC connection"""
        try:
            # Stop worker thread
            self.stop_worker.set()
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
                
            # Close connection
            if self.connection:
                if self.communication_type == 'serial':
                    if self.connection.is_open:
                        self.connection.close()
                elif self.communication_type == 'ethernet':
                    self.connection.close()
                    
            self.connected = False
            logger.info("PLC connection closed")
            
        except Exception as e:
            logger.error(f"Error during PLC disconnect: {e}")
            
    def is_connected(self):
        """Check if PLC connection is active"""
        try:
            if not self.connected or not self.connection:
                return False
                
            if self.communication_type == 'serial':
                return self.connection.is_open
            elif self.communication_type == 'ethernet':
                # Test with a simple read operation
                result = self.connection.read_holding_registers(0, 1)
                return not result.isError()
                
        except Exception:
            return False
            
        return False
        
    def get_queue_size(self):
        """Get current command queue size"""
        return self.command_queue.qsize()
        
    def clear_queue(self):
        """Clear pending commands from queue"""
        try:
            while not self.command_queue.empty():
                self.command_queue.get_nowait()
                self.command_queue.task_done()
                
            logger.info("Command queue cleared")
            
        except Empty:
            pass
        except Exception as e:
            logger.error(f"Error clearing queue: {e}")

# Alternative implementation for direct socket communication
class SocketPLCCommunicator:
    """Alternative PLC communicator using raw TCP sockets"""
    
    def __init__(self, host, port):
        import socket
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect using TCP socket"""
        try:
            import socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(CONNECTION_TIMEOUT)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Socket connection established to {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Socket connection failed: {e}")
            raise
            
    def send_command(self, command):
        """Send command via socket"""
        if not self.connected or not self.socket:
            raise Exception("Socket not connected")
            
        try:
            self.socket.send(command)
            logger.info(f"Command sent via socket: {command}")
            
        except Exception as e:
            logger.error(f"Socket send failed: {e}")
            raise
            
    def disconnect(self):
        """Close socket connection"""
        try:
            if self.socket:
                self.socket.close()
            self.connected = False
            logger.info("Socket connection closed")
            
        except Exception as e:
            logger.error(f"Socket disconnect error: {e}")