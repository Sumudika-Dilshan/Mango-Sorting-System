#!/usr/bin/env python3
"""
Calibration Tool for Mango Sorting System
Interactive tool for calibrating color thresholds and system parameters

Features:
- Interactive HSV threshold adjustment
- Sample image analysis
- Threshold optimization
- Configuration file generation

Author: [Your Name]
Date: [Date]
Version: 1.0
"""

import cv2
import numpy as np
import os
import json
import argparse
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationTool:
    def __init__(self):
        self.current_image = None
        self.current_mask = None
        self.hsv_image = None
        self.window_name = "Mango Calibration Tool"
        self.trackbar_window = "HSV Controls"
        
        # Default thresholds
        self.lower_hsv = [10, 40, 40]
        self.upper_hsv = [40, 255, 255]
        
        # Sample classifications for training
        self.samples = {
            'ripe': [],
            'semiripe': [],
            'unripe': [],
            'defect': []
        }
        
    def setup_trackbars(self):
        """Create trackbar window for HSV adjustment"""
        cv2.namedWindow(self.trackbar_window)
        cv2.resizeWindow(self.trackbar_window, 400, 300)
        
        # Lower HSV trackbars
        cv2.createTrackbar('Lower H', self.trackbar_window, self.lower_hsv[0], 179, self.update_thresholds)
        cv2.createTrackbar('Lower S', self.trackbar_window, self.lower_hsv[1], 255, self.update_thresholds)
        cv2.createTrackbar('Lower V', self.trackbar_window, self.lower_hsv[2], 255, self.update_thresholds)
        
        # Upper HSV trackbars
        cv2.createTrackbar('Upper H', self.trackbar_window, self.upper_hsv[0], 179, self.update_thresholds)
        cv2.createTrackbar('Upper S', self.trackbar_window, self.upper_hsv[1], 255, self.update_thresholds)
        cv2.createTrackbar('Upper V', self.trackbar_window, self.upper_hsv[2], 255, self.update_thresholds)
        
        # Morphology parameters
        cv2.createTrackbar('Morph Size', self.trackbar_window, 7, 20, self.update_thresholds)
        cv2.createTrackbar('Defect Thresh', self.trackbar_window, 50, 100, self.update_thresholds)
        
    def update_thresholds(self, val=None):
        """Update thresholds based on trackbar values"""
        if self.current_image is None:
            return
            
        # Get trackbar values
        self.lower_hsv[0] = cv2.getTrackbarPos('Lower H', self.trackbar_window)
        self.lower_hsv[1] = cv2.getTrackbarPos('Lower S', self.trackbar_window)
        self.lower_hsv[2] = cv2.getTrackbarPos('Lower V', self.trackbar_window)
        
        self.upper_hsv[0] = cv2.getTrackbarPos('Upper H', self.trackbar_window)
        self.upper_hsv[1] = cv2.getTrackbarPos('Upper S', self.trackbar_window)
        self.upper_hsv[2] = cv2.getTrackbarPos('Upper V', self.trackbar_window)
        
        morph_size = cv2.getTrackbarPos('Morph Size', self.trackbar_window)
        if morph_size < 3:
            morph_size = 3
            
        # Apply segmentation
        self.current_mask = self.segment_mango(self.current_image, morph_size)
        
        # Update display
        self.update_display()
        
    def segment_mango(self, image, morph_size=7):
        """Segment mango using current HSV thresholds"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply color threshold
        lower = np.array(self.lower_hsv)
        upper = np.array(self.upper_hsv)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        # Keep largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest_contour], 255)
            
        return mask
        
    def update_display(self):
        """Update the display with current segmentation"""
        if self.current_image is None or self.current_mask is None:
            return
            
        # Create display image
        display = self.current_image.copy()
        
        # Overlay mask
        overlay = np.zeros_like(display)
        overlay[self.current_mask > 0] = [0, 255, 0]  # Green overlay
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        # Add text information
        mask_area = np.sum(self.current_mask > 0)
        total_area = self.current_image.shape[0] * self.current_image.shape[1]
        coverage = mask_area / total_area * 100
        
        cv2.putText(display, f"Coverage: {coverage:.1f}%", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate color statistics
        if mask_area > 0:
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            masked_pixels = hsv[self.current_mask > 0]
            avg_hue = np.mean(masked_pixels[:, 0])
            avg_sat = np.mean(masked_pixels[:, 1])
            
            cv2.putText(display, f"Avg Hue: {avg_hue:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Avg Sat: {avg_sat:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Instructions
        instructions = [
            "Keys: R=Ripe, S=Semi-ripe, U=Unripe, D=Defect",
            "N=Next image, P=Previous, C=Save config, Q=Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display, instruction, (10, display.shape[0] - 40 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(self.window_name, display)
        
    def analyze_color_features(self, image, mask):
        """Extract color features from segmented region"""
        if np.sum(mask) == 0:
            return None
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masked_pixels = hsv[mask > 0]
        
        features = {
            'avg_hue': np.mean(masked_pixels[:, 0]),
            'avg_saturation': np.mean(masked_pixels[:, 1]),
            'avg_value': np.mean(masked_pixels[:, 2]),
            'std_hue': np.std(masked_pixels[:, 0]),
            'std_saturation': np.std(masked_pixels[:, 1])
        }
        
        return features
        
    def add_sample(self, classification):
        """Add current image as training sample"""
        if self.current_image is None or self.current_mask is None:
            return
            
        features = self.analyze_color_features(self.current_image, self.current_mask)
        if features:
            self.samples[classification].append(features)
            logger.info(f"Added {classification} sample. Total: {len(self.samples[classification])}")
            
    def calculate_optimal_thresholds(self):
        """Calculate optimal classification thresholds from samples"""
        if not any(self.samples.values()):
            logger.warning("No samples collected for threshold calculation")
            return None
            
        thresholds = {}
        
        for category, samples in self.samples.items():
            if not samples:
                continue
                
            hues = [s['avg_hue'] for s in samples]
            saturations = [s['avg_saturation'] for s in samples]
            
            thresholds[category] = {
                'hue_min': min(hues) - 2,
                'hue_max': max(hues) + 2,
                'sat_min': min(saturations) - 10,
                'hue_mean': np.mean(hues),
                'sat_mean': np.mean(saturations),
                'sample_count': len(samples)
            }
            
        return thresholds
        
    def save_configuration(self, filename='calibrated_config.json'):
        """Save current configuration to file"""
        config = {
            'timestamp': datetime.now().isoformat(),
            'segmentation_parameters': {
                'lower_hsv': self.lower_hsv,
                'upper_hsv': self.upper_hsv,
                'morph_kernel_size': cv2.getTrackbarPos('Morph Size', self.trackbar_window),
                'defect_threshold_low': cv2.getTrackbarPos('Defect Thresh', self.trackbar_window)
            },
            'color_thresholds': self.calculate_optimal_thresholds(),
            'sample_statistics': {
                category: len(samples) for category, samples in self.samples.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration saved to {filename}")
        
    def load_images(self, image_dir):
        """Load images from directory"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = []
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(supported_formats):
                filepath = os.path.join(image_dir, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    images.append((filepath, image))
                    
        logger.info(f"Loaded {len(images)} images from {image_dir}")
        return images
        
    def run_calibration(self, image_dir):
        """Run interactive calibration process"""
        images = self.load_images(image_dir)
        if not images:
            logger.error("No images found in directory")
            return
            
        self.setup_trackbars()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        current_idx = 0
        
        while True:
            # Load current image
            filepath, image = images[current_idx]
            self.current_image = image.copy()
            self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Apply initial segmentation
            self.update_thresholds()
            
            logger.info(f"Processing: {os.path.basename(filepath)} ({current_idx+1}/{len(images)})")
            
            # Wait for key input
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('n'):  # Next image
                current_idx = (current_idx + 1) % len(images)
            elif key == ord('p'):  # Previous image
                current_idx = (current_idx - 1) % len(images)
            elif key == ord('r'):  # Mark as ripe
                self.add_sample('ripe')
            elif key == ord('s'):  # Mark as semi-ripe
                self.add_sample('semiripe')
            elif key == ord('u'):  # Mark as unripe
                self.add_sample('unripe')
            elif key == ord('d'):  # Mark as defect
                self.add_sample('defect')
            elif key == ord('c'):  # Save configuration
                self.save_configuration()
            elif key == 27:  # Escape
                break
                
        cv2.destroyAllWindows()
        
        # Final configuration save
        self.save_configuration('final_calibration.json')
        
        # Print summary
        print("\n=== Calibration Summary ===")
        for category, samples in self.samples.items():
            print(f"{category.capitalize()}: {len(samples)} samples")
            
        thresholds = self.calculate_optimal_thresholds()
        if thresholds:
            print("\n=== Calculated Thresholds ===")
            for category, params in thresholds.items():
                print(f"{category}: Hue {params['hue_min']:.1f}-{params['hue_max']:.1f}, "
                      f"Sat >= {params['sat_min']:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Mango Sorting System Calibration Tool')
    parser.add_argument('image_dir', help='Directory containing calibration images')
    parser.add_argument('--output', '-o', default='calibration_config.json',
                       help='Output configuration file')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.image_dir):
        logger.error(f"Directory not found: {args.image_dir}")
        return
        
    calibrator = CalibrationTool()
    calibrator.run_calibration(args.image_dir)

if __name__ == "__main__":
    main()