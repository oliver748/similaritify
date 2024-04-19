import sys
import os

sys.path.append(os.path.join("similaritify", "methods"))

import cv2
import time
import utils as ut

class AKAZE:
    def __init__(self, distance_modifier=0.75):
        # dir should be relative to ../image_pairer
        self.detector = cv2.AKAZE_create()
        self.distance_modifier = distance_modifier

    def compare(self, img1, img2):
        # Convert rgb images to grayscale
        gray1, gray2 = ut.convert_to_grayscale(img1, img2)

        # Detect and compute keypoints and descriptors
        kps1, descs1 = ut.find_keypoints_and_descriptors(detector=self.detector, image=gray1)
        kps2, descs2 = ut.find_keypoints_and_descriptors(detector=self.detector, image=gray2)
        
        # If no keypoints or descriptors are found, return 0 (match percentage is 0)
        if any(v is None for v in [descs1, descs2, kps1, kps2]):
            return None

        matcher = ut.create_matcher(norm_type=cv2.NORM_HAMMING)
        
        # Match descriptors
        matches = ut.knn_match_descriptors(matcher, descs1, descs2, k_val=2)

        # Filter valid matches
        valid_matches = ut.find_valid_matches(matches, self.distance_modifier)

        return ut.calculate_match_percentage(valid_matches, kps1)
