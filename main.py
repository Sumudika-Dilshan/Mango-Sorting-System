#!/usr/bin/env python3
"""
Automated Mango Sorting System
Main control script for Raspberry Pi based mango classification and sorting

Hardware Requirements:
- Raspberry Pi 4B
- 2x USB/CSI Cameras
- Photoelectric sensor
- PLC S7-314C 2PN/DP
- Pneumatic cylinders with solenoids

Author: [Your Name]
Date: [Date]
Version: 1.0
"""

import cv2
import numpy as np
import time
import serial
import RPi.GPIO as GPIO
import threading
from datetime import datetime
import logging
from config import *
from image_processor import ImageProcessor
from plc_communication import PLCCommunicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mango_sorting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MangoSortingSystem:
    def __init__(self):
        """Initialize the mango sorting system"""
        self.setup_gpio()
        self.setup_cameras()
        self.image_processor = ImageProcessor()
        self.plc_comm = PLCCommunicator()
        self.running = False
        self.processing_lock = threading.Lock()
        
    def setup_gpio(self):
        """Initialize GPIO pins for sensor input"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(SENSOR_PIN, GPIO.RISING, 
                            callback=self.sensor_callback, 
                            bouncetime=DEBOUNCE_TIME)
        logger.info("GPIO initialized successfully")
        
    def setup_cameras(self):
        """Initialize camera connections"""
        try:
            self.cam1 = cv2.VideoCapture(CAMERA1_INDEX)
            self.cam2 = cv2.VideoCapture(CAMERA2_INDEX)
            
            # Set camera properties
            self.cam1.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cam2.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            
            # Verify camera connections
            if not (self.cam1.isOpened() and self.cam2.isOpened()):
                raise Exception("Failed to initialize cameras")
                
            logger.info("Cameras initialized successfully")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
            
    def sensor_callback(self, channel):
        """Handle photoelectric sensor detection"""
        if not self.processing_lock.acquire(blocking=False):
            logger.warning("Previous mango still processing, skipping detection")
            return
            
        try:
            logger.info("Mango detected by sensor")
            threading.Thread(target=self.process_mango, daemon=True).start()
        except Exception as e:
            logger.error(f"Error in sensor callback: {e}")
            self.processing_lock.release()
            
    def capture_images(self):
        """Capture images from both cameras with timing delays"""
        images = {}
        
        try:
            # Wait for mango to reach Camera 1 position
            time.sleep(CAMERA1_DELAY)
            ret1, frame1 = self.cam1.read()
            
            if ret1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename1 = f"captures/mango_{timestamp}_cam1.jpg"
                cv2.imwrite(filename1, frame1)
                images['camera1'] = frame1
                logger.info(f"Camera 1 image captured: {filename1}")
            
            # Wait for mango to reach Camera 2 position
            time.sleep(CAMERA2_DELAY - CAMERA1_DELAY)
            ret2, frame2 = self.cam2.read()
            
            if ret2:
                filename2 = f"captures/mango_{timestamp}_cam2.jpg"
                cv2.imwrite(filename2, frame2)
                images['camera2'] = frame2
                logger.info(f"Camera 2 image captured: {filename2}")
            
            if not (ret1 and ret2):
                raise Exception("Failed to capture images from both cameras")
                
        except Exception as e:
            logger.error(f"Image capture failed: {e}")
            
        return images
        
    def process_mango(self):
        """Main mango processing workflow"""
        try:
            # Capture images from both cameras
            images = self.capture_images()
            
            if len(images) != 2:
                logger.error("Incomplete image capture")
                return
                
            # Analyze mango using both images
            classification = self.image_processor.analyze_mango(
                images['camera1'], 
                images['camera2']
            )
            
            logger.info(f"Mango classified as: {classification}")
            
            # Send sorting command to PLC
            self.send_sorting_command(classification)
            
            # Log statistics
            self.log_processing_stats(classification)
            
        except Exception as e:
            logger.error(f"Mango processing failed: {e}")
        finally:
            self.processing_lock.release()
            
    def send_sorting_command(self, classification):
        """Send pneumatic actuation command to PLC"""
        try:
            command_map = {
                'Ripe': COMMAND_RIPE,
                'Semi-ripe': COMMAND_SEMIRIPE,
                'Unripe': COMMAND_UNRIPE,
                'Defect': None  # No command for defective mangoes
            }
            
            command = command_map.get(classification)
            
            if command:
                self.plc_comm.send_command(command)
                logger.info(f"Sorting command sent: {command}")
            else:
                logger.info("Defective mango detected - no sorting command sent")
                
        except Exception as e:
            logger.error(f"Failed to send PLC command: {e}")
            
    def log_processing_stats(self, classification):
        """Log processing statistics for monitoring"""
        with open('processing_stats.csv', 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp},{classification}\n")
            
    def start_system(self):
        """Start the sorting system"""
        logger.info("Starting Mango Sorting System...")
        self.running = True
        
        try:
            # Initialize PLC connection
            self.plc_comm.connect()
            
            logger.info("System ready - waiting for mangoes...")
            
            # Main loop
            while self.running:
                time.sleep(0.1)  # Prevent CPU overload
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.shutdown()
            
    def shutdown(self):
        """Clean shutdown of the system"""
        logger.info("Shutting down system...")
        self.running = False
        
        # Release cameras
        if hasattr(self, 'cam1'):
            self.cam1.release()
        if hasattr(self, 'cam2'):
            self.cam2.release()
            
        # Cleanup GPIO
        GPIO.cleanup()
        
        # Close PLC connection
        self.plc_comm.disconnect()
        
        logger.info("System shutdown complete")

def main():
    """Main entry point"""
    try:
        # Create necessary directories
        import os
        os.makedirs('captures', exist_ok=True)
        
        # Initialize and start system
        sorting_system = MangoSortingSystem()
        sorting_system.start_system()
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        
if __name__ == "__main__":
    main()