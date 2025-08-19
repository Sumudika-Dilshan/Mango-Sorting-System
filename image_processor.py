#!/usr/bin/env python3
"""
Image Processing Module for Mango Classification
Handles color analysis, defect detection, and classification logic

Features:
- HSV color space analysis
- Dual-camera image fusion
- Defect detection using morphological operations
- Ripeness classification based on color thresholds

Author: [Your Name]
Date: [Date]
Version: 1.0
"""

import cv2
import numpy as np
import logging
from config import *

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        """Initialize image processor with calibrated thresholds"""
        self.color_thresholds = COLOR_THRESHOLDS
        self.defect_threshold = DEFECT_THRESHOLD
        self.segmentation_params = SEGMENTATION_PARAMS
        
    def analyze_mango(self, image1, image2):
        """
        Analyze mango using both camera images
        
        Args:
            image1 (np.array): Image from camera 1 (top/side view)
            image2 (np.array): Image from camera 2 (opposite view)
            
        Returns:
            str: Classification result ('Ripe', 'Semi-ripe', 'Unripe', 'Defect')
        """
        try:
            # Segment mango in both images
            mask1 = self.segment_mango(image1)
            mask2 = self.segment_mango(image2)
            
            # Extract color features from both images
            color_features1 = self.extract_color_features(image1, mask1)
            color_features2 = self.extract_color_features(image2, mask2)
            
            # Combine color features (weighted average)
            combined_features = self.combine_color_features(
                color_features1, color_features2
            )
            
            # Detect defects in both images
            defects1 = self.detect_defects(image1, mask1)
            defects2 = self.detect_defects(image2, mask2)
            
            # Check for defects first
            total_defect_area = defects1['area'] + defects2['area']
            if total_defect_area > self.defect_threshold:
                logger.info(f"Defect detected - Total area: {total_defect_area}")
                return 'Defect'
            
            # Classify based on color features
            classification = self.classify_ripeness(combined_features)
            
            logger.info(f"Color analysis - Hue: {combined_features['avg_hue']:.2f}, "
                       f"Classification: {classification}")
            
            return classification
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return 'Defect'  # Default to defect for safety
            
    def segment_mango(self, image):
        """
        Segment mango from background using color thresholds
        
        Args:
            image (np.array): Input image
            
        Returns:
            np.array: Binary mask of mango region
        """
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Apply color range mask
            lower_bound = np.array(self.segmentation_params['lower_hsv'])
            upper_bound = np.array(self.segmentation_params['upper_hsv'])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Morphological operations to clean mask
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                self.segmentation_params['morph_kernel_size']
            )
            
            # Close small gaps
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
            
            # Find largest contour (main mango body)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Keep only the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(mask)
                cv2.fillPoly(mask, [largest_contour], 255)
            
            return mask
            
        except Exception as e:
            logger.error(f"Mango segmentation failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
            
    def extract_color_features(self, image, mask):
        """
        Extract color features from masked mango region
        
        Args:
            image (np.array): Input image
            mask (np.array): Binary mask of mango region
            
        Returns:
            dict: Color features (avg_hue, avg_saturation, avg_value, std_hue)
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Extract pixels within mask
            masked_pixels = hsv[mask > 0]
            
            if len(masked_pixels) == 0:
                logger.warning("No pixels found in mask")
                return {
                    'avg_hue': 0, 'avg_saturation': 0, 
                    'avg_value': 0, 'std_hue': 0
                }
            
            # Calculate statistics
            features = {
                'avg_hue': np.mean(masked_pixels[:, 0]),
                'avg_saturation': np.mean(masked_pixels[:, 1]),
                'avg_value': np.mean(masked_pixels[:, 2]),
                'std_hue': np.std(masked_pixels[:, 0])
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Color feature extraction failed: {e}")
            return {'avg_hue': 0, 'avg_saturation': 0, 'avg_value': 0, 'std_hue': 0}
            
    def combine_color_features(self, features1, features2):
        """
        Combine color features from both cameras
        
        Args:
            features1 (dict): Features from camera 1
            features2 (dict): Features from camera 2
            
        Returns:
            dict: Combined features
        """
        combined = {}
        
        for key in features1.keys():
            # Weighted average (can be adjusted based on camera importance)
            combined[key] = (features1[key] * 0.5 + features2[key] * 0.5)
            
        return combined
        
    def detect_defects(self, image, mask):
        """
        Detect defects like dark spots, bruises, or abnormal areas
        
        Args:
            image (np.array): Input image
            mask (np.array): Mango segmentation mask
            
        Returns:
            dict: Defect information (area, count, severity)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply mask to focus only on mango region
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Detect dark spots (potential defects)
            _, defect_mask = cv2.threshold(
                masked_gray, 
                self.segmentation_params['defect_threshold_low'], 
                255, 
                cv2.THRESH_BINARY_INV
            )
            
            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
            
            # Find defect contours
            contours, _ = cv2.findContours(
                defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Calculate defect metrics
            total_defect_area = sum(cv2.contourArea(cnt) for cnt in contours)
            defect_count = len([cnt for cnt in contours 
                              if cv2.contourArea(cnt) > self.segmentation_params['min_defect_area']])
            
            # Calculate severity based on area ratio
            total_mango_area = np.sum(mask > 0)
            severity = total_defect_area / total_mango_area if total_mango_area > 0 else 0
            
            return {
                'area': total_defect_area,
                'count': defect_count,
                'severity': severity
            }
            
        except Exception as e:
            logger.error(f"Defect detection failed: {e}")
            return {'area': 0, 'count': 0, 'severity': 0}
            
    def classify_ripeness(self, color_features):
        """
        Classify mango ripeness based on color features
        
        Args:
            color_features (dict): Combined color features
            
        Returns:
            str: Ripeness classification
        """
        avg_hue = color_features['avg_hue']
        avg_saturation = color_features['avg_saturation']
        
        # Classification based on calibrated thresholds
        thresholds = self.color_thresholds
        
        # Check ripeness categories in order of priority
        if (thresholds['ripe']['hue_min'] <= avg_hue <= thresholds['ripe']['hue_max'] and
            avg_saturation >= thresholds['ripe']['sat_min']):
            return 'Ripe'
            
        elif (thresholds['semiripe']['hue_min'] <= avg_hue <= thresholds['semiripe']['hue_max'] and
              avg_saturation >= thresholds['semiripe']['sat_min']):
            return 'Semi-ripe'
            
        elif (avg_hue <= thresholds['unripe']['hue_max'] and
              avg_saturation >= thresholds['unripe']['sat_min']):
            return 'Unripe'
            
        else:
            logger.warning(f"Unclassified color - Hue: {avg_hue}, Sat: {avg_saturation}")
            return 'Defect'  # Default to defect for unclassified mangoes
            
    def save_debug_images(self, image, mask, defect_mask, classification, timestamp):
        """
        Save debug images for analysis and calibration
        
        Args:
            image (np.array): Original image
            mask (np.array): Segmentation mask
            defect_mask (np.array): Defect detection mask
            classification (str): Classification result
            timestamp (str): Timestamp for filename
        """
        try:
            import os
            debug_dir = 'debug_images'
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save original with overlay
            overlay = image.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
            overlay[defect_mask > 0] = overlay[defect_mask > 0] * 0.7 + np.array([0, 0, 255]) * 0.3
            
            filename = f"{debug_dir}/debug_{timestamp}_{classification}.jpg"
            cv2.imwrite(filename, overlay)
            
            logger.debug(f"Debug image saved: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save debug image: {e}")