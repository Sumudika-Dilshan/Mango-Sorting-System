#!/usr/bin/env python3
"""
Configuration file for Mango Sorting System
Contains all system parameters, thresholds, and hardware settings

Author: [Your Name]
Date: [Date]
Version: 1.0
"""

# GPIO Configuration
SENSOR_PIN = 18              # Photoelectric sensor input pin
DEBOUNCE_TIME = 500          # Sensor debounce time in ms

# Camera Configuration
CAMERA1_INDEX = 0            # USB Camera 1 (top/side view)
CAMERA2_INDEX = 1            # USB Camera 2 (opposite view)
CAMERA_WIDTH = 640           # Camera resolution width
CAMERA_HEIGHT = 480          # Camera resolution height

# Timing Configuration (in seconds)
CAMERA1_DELAY = 2.0          # Delay after sensor trigger to capture camera 1
CAMERA2_DELAY = 3.0          # Delay after sensor trigger to capture camera 2

# PLC Communication Configuration
PLC_PORT = '/dev/ttyUSB0'    # Serial port for PLC communication
PLC_BAUDRATE = 9600          # Serial communication baud rate
PLC_TIMEOUT = 1.0            # Communication timeout in seconds

# PLC Commands
COMMAND_RIPE = b'ACT1\n'     # Command to activate pneumatic cylinder 1 (ripe)
COMMAND_SEMIRIPE = b'ACT2\n' # Command to activate pneumatic cylinder 2 (semi-ripe)
COMMAND_UNRIPE = b'ACT3\n'   # Command to activate pneumatic cylinder 3 (unripe)

# Image Processing Configuration
DEFECT_THRESHOLD = 560       # Maximum allowable defect area in pixels

# Color Classification Thresholds (HSV values)
COLOR_THRESHOLDS = {
    'ripe': {
        'hue_min': 20,           # Minimum hue for ripe mangoes
        'hue_max': 35,           # Maximum hue for ripe mangoes
        'sat_min': 80            # Minimum saturation for ripe mangoes
    },
    'semiripe': {
        'hue_min': 36,           # Minimum hue for semi-ripe mangoes
        'hue_max': 50,           # Maximum hue for semi-ripe mangoes
        'sat_min': 70            # Minimum saturation for semi-ripe mangoes
    },
    'unripe': {
        'hue_max': 19,           # Maximum hue for unripe mangoes
        'sat_min': 60            # Minimum saturation for unripe mangoes
    }
}

# Segmentation Parameters
SEGMENTATION_PARAMS = {
    'lower_hsv': [10, 40, 40],   # Lower bound for mango color in HSV
    'upper_hsv': [40, 255, 255], # Upper bound for mango color in HSV
    'morph_kernel_size': (7, 7), # Morphological operation kernel size
    'defect_threshold_low': 50,   # Lower threshold for defect detection
    'min_defect_area': 10        # Minimum area to consider as defect
}

# System Performance Parameters
MAX_PROCESSING_TIME = 5.0    # Maximum time allowed for processing one mango (seconds)
CAPTURE_RETRY_COUNT = 3      # Number of retries for image capture
LOG_LEVEL = 'INFO'           # Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

# File Paths
LOG_FILE = 'mango_sorting.log'
STATS_FILE = 'processing_stats.csv'
CAPTURE_DIR = 'captures'
DEBUG_DIR = 'debug_images'

# Calibration Parameters (adjust based on your setup)
CONVEYOR_SPEED = 0.5         # Conveyor belt speed in m/s (for timing calculations)
CAMERA_DISTANCE = 0.3        # Distance between cameras in meters
SENSOR_TO_CAMERA1 = 0.8      # Distance from sensor to camera 1 in meters
CAMERA1_TO_CAMERA2 = 0.3     # Distance from camera 1 to camera 2 in meters

# Quality Control Parameters
MIN_MANGO_AREA = 1000        # Minimum pixel area to consider as valid mango
MAX_MANGO_AREA = 50000       # Maximum pixel area to consider as valid mango
CONFIDENCE_THRESHOLD = 0.7   # Minimum confidence for classification

# Network Configuration (if using Ethernet PLC communication)
PLC_IP_ADDRESS = '192.168.1.100'  # PLC IP address for Ethernet communication
PLC_PORT_NUMBER = 502              # Modbus TCP port
CONNECTION_TIMEOUT = 5.0           # Network connection timeout